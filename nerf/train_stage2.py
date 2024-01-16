import datetime
import os

import nvdiffrast.torch as dr
import torch
import wandb
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
import render.renderutils as ru
from dataset.nerf_synthetic import NerfSyntheticDataset
from dataset.tensoir_synthetic import TensoirSyntheticDataset
from render import light, material, mlptexture, util
from utils import *


def load_config():
    register_operation()

    valid_models = ['TensorCP', 'TensorVM', 'TensorVMSplit', 'NeRF', 'NeuS', 'TensoIR']
    valid_datasets = ['nerf_synthetic', 'tensoir_synthetic']
    general_conf = OmegaConf.load('config/general_second.yaml')
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(general_conf, cli_conf)

    if conf.data.name in valid_datasets:
        data_conf = OmegaConf.load(f'config/{conf.data.name}/{conf.data.type}.yaml')
    else:
        raise NotImplementedError('Unknown dataset type: %s' % conf.data.name)

    if conf.model.geo_model.name in valid_models:
        geo_conf = OmegaConf.load(f'config/model/{conf.model.geo_model.name}.yaml')
    else:
        raise NotImplementedError('Unknown geometry model: %s' % conf.model.geo_model.name)

    if conf.model.app_model.name in valid_models:
        app_conf = OmegaConf.load(f'config/model/{conf.model.app_model.name}.yaml')
    else:
        raise NotImplementedError('Unknown appearance model: %s' % conf.model.app_model.name)

    conf = OmegaConf.merge(data_conf, general_conf)
    conf.model.geo_model = OmegaConf.merge(geo_conf.model, conf.model.geo_model)
    conf.model.app_model = OmegaConf.merge(app_conf.model, conf.model.app_model)

    # ensure all config input from command line are valid
    def search_conf(nw_conf, config):
        for k, v in config.items():
            print(k)
            if k not in nw_conf:
                raise ValueError(f'Unknown config: {k}')
            if hasattr(v, 'keys'):
                search_conf(nw_conf[k], v)

    search_conf(conf, cli_conf)
    conf = OmegaConf.merge(conf, cli_conf)

    print('==================> Configurations <==================')
    print(OmegaConf.to_yaml(conf), end='')
    print('======================================================')
    return conf


def load_data(data_conf, need_test=False, is_stack=False):
    TRAIN_DATASET = None
    VAL_DATASET = None
    TEST_DATASET = None
    print(data_conf)

    if data_conf.name == 'nerf_synthetic':
        TRAIN_DATASET = NerfSyntheticDataset(
            datadir=data_conf.dir,
            split='train',
            downsample=data_conf.downsample,
            is_stack=is_stack,
            crop=data_conf.crop_train,
            debug=data_conf.debug,
        )
        VAL_DATASET = NerfSyntheticDataset(
            datadir=data_conf.dir,
            split='val',
            downsample=data_conf.downsample,
            is_stack=True,
            crop=data_conf.crop_val,
            debug=data_conf.debug,
        )
        if need_test:
            TEST_DATASET = NerfSyntheticDataset(
                datadir=data_conf.dir,
                split='test',
                downsample=data_conf.downsample,
                is_stack=True,
                crop=data_conf.crop_val,
            debug=data_conf.debug,
            )
    elif data_conf.name == 'tensoir_synthetic':
        TRAIN_DATASET = TensoirSyntheticDataset(
            datadir=data_conf.dir,
            split='train',
            downsample=data_conf.downsample,
            is_stack=is_stack,
            crop=data_conf.crop_train,
            debug=data_conf.debug,
        )
        VAL_DATASET = TensoirSyntheticDataset(
            datadir=data_conf.dir,
            split='val',
            downsample=data_conf.downsample,
            is_stack=True,
            crop=data_conf.crop_val,
            debug=data_conf.debug,
        )
        if need_test:
            TEST_DATASET = TensoirSyntheticDataset(
                datadir=data_conf.dir,
                split='test',
                downsample=data_conf.downsample,
                is_stack=True,
                crop=data_conf.crop_val,
                debug=data_conf.debug,
            )
    else:
        raise NotImplementedError('Unknown dataset type: %s' % data_conf.name)

    return TRAIN_DATASET, VAL_DATASET, TEST_DATASET


def set_device(gpu):
    if gpu == -1:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda:%d' % gpu)
    else:
        device = torch.device('cpu')
        print('Warning: CPU is used because GPU is not available.')
    return device


def initial_guess_material(geometry, FLAGS, base_mesh=None, init_mat=None, tensorf_model=None):
    if FLAGS.tex_type == "neu_diffrec":
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=FLAGS.tex_dim)
    elif FLAGS.tex_type == "mlp":
        mlp_map_opt = mlptexture.MLPNeuralTex(geometry.getAABB(), channels=FLAGS.tex_dim, pospe=FLAGS.pospe, feape=FLAGS.feape, viewpe=FLAGS.viewpe)
    elif FLAGS.tex_type == "vert" and base_mesh is not None:
        mlp_map_opt = mlptexture.VertNeuralTex(geometry.getAABB(), n_verts=base_mesh.v_pos.shape[0], channels=FLAGS.tex_dim, feape=FLAGS.feape, viewpe=FLAGS.viewpe)
    elif FLAGS.tex_type == "vert_sh" and base_mesh is not None:
        mlp_map_opt = mlptexture.VertSHTex(n_verts=base_mesh.v_pos.shape[0], deg=FLAGS.deg, channels=FLAGS.tex_dim, feape=FLAGS.feape, viewpe=FLAGS.viewpe)
    elif FLAGS.tex_type == "tensorVM":
        mlp_map_opt = mlptexture.TensorVMSplitNeuralTex(geometry.getAABB(), channels=FLAGS.tex_dim, pospe=FLAGS.pospe, feape=FLAGS.feape, viewpe=FLAGS.viewpe, shader_internal_dims=FLAGS.shader_internal_dims)
    elif FLAGS.tex_type == "tensorVM_SH":
        mlp_map_opt = mlptexture.TensorSHNeuralTex(geometry.getAABB(), channels=FLAGS.tex_dim, pospe=FLAGS.pospe, feape=FLAGS.feape, viewpe=FLAGS.viewpe, shader_internal_dims=FLAGS.shader_internal_dims)
    elif FLAGS.tex_type == "tensorVM_preload" and tensorf_model is not None:
        mlp_map_opt = mlptexture.TensorVMSplitLoadNeuralTex(tensorf_model, channels=FLAGS.tex_dim, unbounded=FLAGS.unbounded)
    elif FLAGS.tex_type == "neus_preload" and tensorf_model is not None:
        mlp_map_opt = mlptexture.NeuSLoadNeuralTex(tensorf_model, channels=FLAGS.tex_dim)
    elif FLAGS.tex_type == "uvmap":
        mlp_map_opt = mlptexture.UVMapNeuralTex(channels=FLAGS.tex_dim, pospe=FLAGS.pospe, feape=FLAGS.feape, viewpe=FLAGS.viewpe)
    elif FLAGS.tex_type == "tensoIR_physical":
        mlp_map_opt = mlptexture.TensoIRPhysicalRendering(tensorf_model, channels=FLAGS.tex_dim, unbounded=FLAGS.unbounded)
    else:
        raise ValueError("Texture Type ERROR! Texture Type: {}, Base Mesh: {}".format(FLAGS.tex_type, base_mesh))

    mat = material.Material({'neural_tex': mlp_map_opt})

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    return mat


@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False


@torch.no_grad()
def prepare_batch(target, args):
    reso = [int(args.resolution[0] // args.downsample), int(args.resolution[1] // args.downsample)]
    if not args.anti_aliasing:
        reso[0] = target['img'].shape[1]
        reso[1] = target['img'].shape[2]
    if args.bg_type == 'checker':
        background = torch.tensor(util.checkerboard(reso, 8), dtype=torch.float32)[None, ...]
    elif args.bg_type == 'black':
        background = torch.zeros((1, reso[0], reso[1], 3), dtype=torch.float32)
    elif args.bg_type == 'white':
        background = torch.ones((1, reso[0], reso[1], 3), dtype=torch.float32)
    elif args.bg_type == 'random':
        background = torch.rand((1, reso[0], reso[1], 3), dtype=torch.float32)
    else:
        assert False, "Unknown background type %s" % args.bg_type

    target['resolution'] = reso
    target['background'] = background

    if args.anti_aliasing:
        background = torch.nn.functional.interpolate(background.permute(0,3,1,2), scale_factor=[1/args.downsample, 1/args.downsample], mode=args.anti_aliasing_mode, align_corners=True).permute(0,2,3,1)
    if target['img'].shape[-1] == 4:
        target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)

    return target


@torch.no_grad()
def validate_itr(glctx, target, geometry, opt_material, FLAGS):
    result_dict = {}
    with torch.no_grad():
        st = time.time()
        buffers = geometry.render(glctx, target, opt_material)
        et = time.time()
        result_dict["time"] = et - st

        # print(target['img'].shape)

        # result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        # result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_dict['ref'] = target['img'][...,0:3][0]
        result_dict['opt'] = buffers['shaded'][...,0:3][0]
        result_dict['full_ref'] = target['img'][0]
        result_dict['full_opt'] = buffers['shaded'][0]
        result_dict['wo_indir_rgb'] = buffers['wo_indir_rgb']
        result_dict['direct_rgb'] = buffers['wo_visibility_direct_rgb']
        result_dict['albedo'] = buffers['albedo']
        result_dict['roughness'] = buffers['roughness']
        result_dict['indirect_rgb'] = buffers['indirect_light_rgb']
        result_dict['normal'] = buffers['normal'] * 0.5 + 0.5
        result_dict['pos'] = buffers['gb_pos']
        result_dict['positions'] = buffers['positions']

        # print(buffers['normal'].max(), buffers['normal'].min())

        if result_dict['opt'].shape == result_dict['ref'].shape:
            result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)
        else:
            result_image = result_dict['opt']

        return result_image, result_dict


def run_validate(glctx, geometry, opt_material, dataset_validate, out_dir, FLAGS, device):
    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []

    total_time = 0

    dataloader_validate = DataLoader(
        dataset_validate,
        batch_size=1,
    )

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')

        total_time = 0
        print("Running validation")
        for it, nwdata in enumerate(dataloader_validate):
            # Mix validation background
            for i in range(FLAGS.data.downsample):
                for j in range(FLAGS.data.downsample):
                    cx = dataset_validate.w // FLAGS.data.downsample
                    cy = dataset_validate.h // FLAGS.data.downsample
                    cx = cx // 2 + i * cx
                    cy = cy // 2 + j * cy
                    target = prepare_batch(dataset_validate.get_crop(nwdata, cx, cy), FLAGS.data)
                    target['spp'] = 1
                    for k, v in target.items():
                        if isinstance(v, torch.Tensor):
                            target[k] = v.to(device)

                    # print("Validating %d/%d" % (it, len(dataset_validate)))
                    # print("cx: %d, cy: %d" % (cx, cy))
                    # nw_img = target['img'][0].detach().cpu().numpy()
                    # print(nw_img.shape) # (256, 256, 4)
                    # # blend alpha channel
                    # nw_img = nw_img[..., 0:3] * nw_img[..., 3:4]
                    # nw_img = np.clip(nw_img, 0.0, 1.0)
                    # import imageio
                    # imageio.imwrite("test.png", np.uint8(nw_img*255))


                    FLAGS.display = None

                    result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, FLAGS)

                    # result_image = result_image.detach().cpu().numpy()
                    # im = Image.fromarray(np.uint8(result_image*255))
                    # im.save(out_dir + '/' + ('val_%06d.png' % it))

                    total_time += result_dict['time']
                    # Compute metrics
                    opt = torch.clamp(result_dict['opt'], 0.0, 1.0)
                    ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

                    mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
                    mse_values.append(float(mse))
                    psnr = util.mse_to_psnr(mse)
                    psnr_values.append(float(psnr))

                    line = "%d, %1.8f, %1.8f\n" % (it, mse, psnr)
                    fout.write(str(line))

                    for k in result_dict.keys():
                        if k != "time":
                            np_img = result_dict[k].detach().cpu().numpy()
                            util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
        fout.write(str(line))
        print("MSE,      PSNR")
        print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
        print("Total inference time:", total_time)

    return avg_psnr


def run_validate_crop(glctx, geometry, opt_material, dataset_validate, out_dir, FLAGS, device):
    mse_values = []
    psnr_values = []

    total_time = 0

    dataloader_validate = DataLoader(
        dataset_validate,
        batch_size=1,
    )
    print('validate downsample', dataset_validate.downsample)
    os.makedirs(out_dir, exist_ok=True)

    from render import obj
    opt_mesh = geometry.getMesh(opt_material)
    obj.write_ply(out_dir, opt_mesh, save_material=False)


    _, view_dirs = opt_mesh.material['neural_tex'].net.generate_envir_map_dir(256, 512)

    predicted_envir_map = opt_mesh.material['neural_tex'].net.get_light_rgbs(view_dirs.reshape(-1, 3).to(device))[0]
    predicted_envir_map = predicted_envir_map.reshape(256, 512, 3).cpu().detach().numpy()
    predicted_envir_map = np.clip(predicted_envir_map, a_min=0, a_max=np.inf)
    predicted_envir_map = np.uint8(np.clip(np.power(predicted_envir_map, 1./2.2), 0., 1.) * 255.)
    envirmap = predicted_envir_map

    # save predicted envir map
    import imageio
    imageio.imwrite(f'{out_dir}/envirmap.png', envirmap)

    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')

        total_time = 0
        print("Running validation")

        tx = dataset_validate.w // FLAGS.data.downsample
        ty = dataset_validate.h // FLAGS.data.downsample

        for it, nwdata in enumerate(dataloader_validate):
            # Mix validation background

            my_img = np.zeros((dataset_validate.h, dataset_validate.w, 3), dtype=np.float32)
            ref_img = np.zeros((dataset_validate.h, dataset_validate.w, 3), dtype=np.float32)
            albedo_img = np.zeros((dataset_validate.h, dataset_validate.w, 3), dtype=np.float32)
            roughness_img = np.zeros((dataset_validate.h, dataset_validate.w, 3), dtype=np.float32)
            my_full_img = np.zeros((dataset_validate.h, dataset_validate.w, 4), dtype=np.float32)
            ref_full_img = np.zeros((dataset_validate.h, dataset_validate.w, 4), dtype=np.float32)
            wo_indir_img = np.zeros((dataset_validate.h, dataset_validate.w, 3), dtype=np.float32)
            direct_rgb_img = np.zeros((dataset_validate.h, dataset_validate.w, 3), dtype=np.float32)
            indirect_rgb_img = np.zeros((dataset_validate.h, dataset_validate.w, 3), dtype=np.float32)
            normal_img = np.zeros((dataset_validate.h, dataset_validate.w, 3), dtype=np.float32)
            positions_img = np.zeros((dataset_validate.h, dataset_validate.w, 3), dtype=np.float32)
            gt_pos_img = np.zeros((dataset_validate.h, dataset_validate.w, 3), dtype=np.float32)

            for i in range(FLAGS.data.downsample):
                for j in range(FLAGS.data.downsample):
                    cx = tx // 2 + i * tx
                    cy = ty // 2 + j * ty
                    target = prepare_batch(dataset_validate.get_crop(nwdata, cx, cy), FLAGS.data)
                    target['spp'] = 1
                    for k, v in target.items():
                        if isinstance(v, torch.Tensor):
                            target[k] = v.to(device)

                    print("Validating %d/%d" % (it, len(dataset_validate)))
                    print("cx: %d, cy: %d" % (cx, cy))

                    FLAGS.display = None

                    _, result_dict = validate_itr(glctx, target, geometry, opt_material, FLAGS)

                    # total_time += result_dict['time']
                    # Compute metrics
                    opt = torch.clamp(result_dict['opt'], 0.0, 1.0)
                    ref = torch.clamp(result_dict['ref'], 0.0, 1.0)
                    # mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()

                    my_img[i * tx:(i + 1) * tx, j * ty:(j + 1) * ty, :] = opt.detach().cpu().numpy()
                    ref_img[i * tx:(i + 1) * tx, j * ty:(j + 1) * ty, :] = ref.detach().cpu().numpy()
                    albedo_img[i * tx:(i + 1) * tx, j * ty:(j + 1) * ty, :] = result_dict['albedo'].detach().cpu().numpy()
                    roughness = result_dict['roughness'].detach().cpu().numpy() # (H, W, 1)
                    roughness_img[i * tx:(i + 1) * tx, j * ty:(j + 1) * ty, :] = np.repeat(roughness, 3, axis=2)

                    my_full_img[i * tx:(i + 1) * tx, j * ty:(j + 1) * ty, :] = result_dict['full_opt'].detach().cpu().numpy()
                    ref_full_img[i * tx:(i + 1) * tx, j * ty:(j + 1) * ty, :] = result_dict['full_ref'].detach().cpu().numpy()
                    wo_indir_img[i * tx:(i + 1) * tx, j * ty:(j + 1) * ty, :] = result_dict['wo_indir_rgb'].detach().cpu().numpy()
                    direct_rgb_img[i * tx:(i + 1) * tx, j * ty:(j + 1) * ty, :] = result_dict['direct_rgb'].detach().cpu().numpy()
                    indirect_rgb_img[i * tx:(i + 1) * tx, j * ty:(j + 1) * ty, :] = result_dict['indirect_rgb'].detach().cpu().numpy()
                    normal_img[i * tx:(i + 1) * tx, j * ty:(j + 1) * ty, :] = result_dict['normal'].detach().cpu().numpy()

            util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, "my_img")), my_img)
            util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, "ref_img")), ref_img)
            util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, "albedo_img")), albedo_img)
            util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, "roughness_img")), roughness_img)

            util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, "my_full_img")), my_full_img)
            util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, "ref_full_img")), ref_full_img)
            util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, "wo_indir_img")), wo_indir_img)
            util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, "direct_rgb_img")), direct_rgb_img)
            util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, "indirect_rgb_img")), indirect_rgb_img)
            util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, "normal_img")), normal_img)

            # mse = torch.nn.functional.mse_loss(my_img, ref_img, size_average=None, reduce=None, reduction='mean').item()
            mse = np.mean(np.square(my_img - ref_img))
            mse_values.append(float(mse))
            psnr = util.mse_to_psnr(mse)
            psnr_values.append(float(psnr))
            line = "%d, %1.8f, %1.8f\n" % (it, mse, psnr)
            fout.write(str(line))

        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
        fout.write(str(line))
        print("MSE,      PSNR")
        print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
        print("Total inference time:", total_time)

    return avg_psnr


def test(args):
    if args.wandb:
        wandb.login()
        wandb.init(
            project='3d_reconstruction_all',
            config=args,
            name=args.exp+'-test',
            notes=args.notes,
            tags=args.tags,
        )
        wandb.config.update(args)

    seed_everything(args.seed)
    torch.set_default_dtype(torch.float32)

    logdir = (f'{args.logdir}/{args.exp}/'
        f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
    print('Logdir: %s' % logdir)
    os.makedirs(logdir, exist_ok=True)
    device = set_device(args.gpu)

    TRAIN_DATASET, VAL_DATASET, TEST_DATASET = load_data(args.data, need_test=True, is_stack=True)

    aabb = TRAIN_DATASET.scene_bbox
    args.model.geo_model.near_far = TRAIN_DATASET.near_far
    args.model.app_model.near_far = TRAIN_DATASET.near_far
    args.model.geo_model.white_bg = TRAIN_DATASET.white_bg
    args.model.app_model.white_bg = TRAIN_DATASET.white_bg
    grid_size = N_to_reso(getattr(args, 'N_voxel_init', 2097152), aabb)

    if hasattr(args.model.geo_model, 'ckpt'):
        ckpt = torch.load(args.model.geo_model.ckpt, map_location=device)
        geo_conf = args.model.geo_model
        if 'kwargs' in ckpt.keys():
            aabb = ckpt['kwargs']['aabb']
            grid_size = ckpt['kwargs']['grid_size']
            geo_conf.density.n_comp = ckpt['kwargs']['density_n_comp']
            geo_conf.app.n_comp = ckpt['kwargs']['app_n_comp']
            geo_conf.app.feature_dim = ckpt['kwargs']['app_dim']
            geo_conf.near_far = ckpt['kwargs']['near_far']
            geo_conf.step_ratio = ckpt['kwargs']['step_ratio']
        geo_model = eval('models.' + args.model.geo_model.name)(geo_conf, device, aabb, grid_size).to(device)

        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(
                ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            geo_model.alphaMask = models.AlphaGridMask(
                device, ckpt['alphaMask.aabb'], alpha_volume.float().to(device))
        geo_model.load_state_dict(ckpt['state_dict'])
    else:
        geo_model = eval('models.' + args.model.geo_model.name)(args.model.geo_model, device, aabb, grid_size).to(device)

    if hasattr(args.model.app_model, 'ckpt'):
        ckpt = torch.load(args.model.app_model.ckpt, map_location=device)
        app_conf = args.model.app_model
        if 'kwargs' in ckpt.keys():
            print(ckpt['kwargs'])
            aabb = ckpt['kwargs']['aabb']
            grid_size = ckpt['kwargs']['grid_size']
            app_conf.density.n_comp = ckpt['kwargs']['density_n_comp']
            app_conf.app.n_comp = ckpt['kwargs']['app_n_comp']
            app_conf.app.feature_dim = ckpt['kwargs']['app_dim']
            app_conf.near_far = ckpt['kwargs']['near_far']
            app_conf.step_ratio = ckpt['kwargs']['step_ratio']
        app_model = eval('models.' + args.model.app_model.name)(args.model.app_model, device, aabb, grid_size).to(device)

        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(
                ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            app_model.alphaMask = models.AlphaGridMask(
                device, ckpt['alphaMask.aabb'], alpha_volume.float().to(device))
        app_model.load_state_dict(ckpt['state_dict'])
    else:
        app_model = eval('models.' + args.model.app_model.name)(args.model.app_model, device, aabb, grid_size).to(device)

    app_model.is_relight=True

    geo_model_dmtet_name = args.model.geo_model.name + '_DMTet'
    geo_dmtet = eval('models.' + geo_model_dmtet_name)(geo_model, device, args.model).to(device)

    mat = initial_guess_material(geo_dmtet, args.model, tensorf_model=app_model)
    glctx = dr.RasterizeCudaContext()

    ckpt_pth = args.ckpt
    ckpt = torch.load(ckpt_pth, map_location=device)
    geo_dmtet.load_state_dict(ckpt['geo_dmtet'])
    mat['neural_tex'].load_state_dict(ckpt['neural_tex_state_dict'])

    psnr_test = run_validate_crop(glctx, geo_dmtet, mat, TEST_DATASET, os.path.join(logdir, "test"), args, device)
    if args.wandb:
        wandb.log({'psnr_test': psnr_test})
    else:
        print(f'psnr_test: {psnr_test}')


def train(args):
    if args.wandb:
        wandb.login()
        wandb.init(
            project='3d_reconstruction_all',
            config=args,
            name=args.exp,
            notes=args.notes,
            tags=args.tags,
        )
        wandb.config.update(args)

    seed_everything(args.seed)
    torch.set_default_dtype(torch.float32)

    logdir = (f'{args.logdir}/{args.exp}/'
        f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
    print('Logdir: %s' % logdir)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(f'{logdir}/imgs_vis', exist_ok=True)

    # backup source code
    if not args.no_backup:
        backup(logdir, args)

    device = set_device(args.gpu)

    TRAIN_DATASET, VAL_DATASET, TEST_DATASET = load_data(args.data, args.render_test, is_stack=True)

    train_loader = DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    train_iter = iter(cycle(train_loader))

    aabb = TRAIN_DATASET.scene_bbox
    args.model.geo_model.near_far = TRAIN_DATASET.near_far
    args.model.app_model.near_far = TRAIN_DATASET.near_far
    args.model.geo_model.white_bg = TRAIN_DATASET.white_bg
    args.model.app_model.white_bg = TRAIN_DATASET.white_bg
    grid_size = N_to_reso(getattr(args, 'N_voxel_init', 2097152), aabb)

    if hasattr(args.model.geo_model, 'ckpt'):
        ckpt = torch.load(args.model.geo_model.ckpt, map_location=device)
        geo_conf = args.model.geo_model
        if 'kwargs' in ckpt.keys():
            aabb = ckpt['kwargs']['aabb']
            grid_size = ckpt['kwargs']['grid_size']
            geo_conf.density.n_comp = ckpt['kwargs']['density_n_comp']
            geo_conf.app.n_comp = ckpt['kwargs']['app_n_comp']
            geo_conf.app.feature_dim = ckpt['kwargs']['app_dim']
            geo_conf.near_far = ckpt['kwargs']['near_far']
            geo_conf.step_ratio = ckpt['kwargs']['step_ratio']
        geo_model = eval('models.' + args.model.geo_model.name)(geo_conf, device, aabb, grid_size).to(device)

        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(
                ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            geo_model.alphaMask = models.AlphaGridMask(
                device, ckpt['alphaMask.aabb'], alpha_volume.float().to(device))
        geo_model.load_state_dict(ckpt['state_dict'])
    else:
        geo_model = eval('models.' + args.model.geo_model.name)(args.model.geo_model, device, aabb, grid_size).to(device)

    if hasattr(args.model.app_model, 'ckpt'):
        ckpt = torch.load(args.model.app_model.ckpt, map_location=device)
        app_conf = args.model.app_model
        if 'kwargs' in ckpt.keys():
            print(ckpt['kwargs'])
            aabb = ckpt['kwargs']['aabb']
            grid_size = ckpt['kwargs']['grid_size']
            app_conf.density.n_comp = ckpt['kwargs']['density_n_comp']
            app_conf.app.n_comp = ckpt['kwargs']['app_n_comp']
            app_conf.app.feature_dim = ckpt['kwargs']['app_dim']
            app_conf.near_far = ckpt['kwargs']['near_far']
            app_conf.step_ratio = ckpt['kwargs']['step_ratio']
        app_model = eval('models.' + args.model.app_model.name)(args.model.app_model, device, aabb, grid_size).to(device)

        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(
                ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            app_model.alphaMask = models.AlphaGridMask(
                device, ckpt['alphaMask.aabb'], alpha_volume.float().to(device))
        app_model.load_state_dict(ckpt['state_dict'])
    else:
        app_model = eval('models.' + args.model.app_model.name)(args.model.app_model, device, aabb, grid_size).to(device)

    app_model.is_relight=True

    geo_model_dmtet_name = args.model.geo_model.name + '_DMTet'
    geo_dmtet = eval('models.' + geo_model_dmtet_name)(geo_model, device, args.model).to(device)

    total_params = sum(p.numel() for p in geo_dmtet.parameters())
    print("Total number of parameters: {}M".format(total_params/1024/1024))

    mat = initial_guess_material(geo_dmtet, args.model, tensorf_model=app_model)
    glctx = dr.RasterizeCudaContext()

    optim_dict = {
        'Adam': torch.optim.Adam,
    }
    grad_vars = [
        {'params': filter(lambda p: p.requires_grad, mat.parameters()), 'lr': args.optimizer.lr_mat},
        {'params': filter(lambda p: p.requires_grad, geo_dmtet.parameters()), 'lr': args.optimizer.lr_pos},
    ]

    optimizer = optim_dict[args.optimizer.name](grad_vars, **args.optimizer.params)
    scheduler = build_scheduler(optimizer, args.scheduler)

    if args.exp == 'use_derived_normal':
        mat['neural_tex'].net.normals_kind = 'purely_derived'

    loss_fn = createLoss(args)

    psnr_test = run_validate_crop(glctx, geo_dmtet, mat, VAL_DATASET, os.path.join(logdir, "validate_ini"), args, device)
    if args.wandb:
        wandb.log({'psnr_test': psnr_test})
    else:
        print(f'psnr_test: {psnr_test}')

    for i in tqdm(range(args.iteration)):
        args.nw_iter = i
        data = next(train_iter)
        cx = np.random.randint(100, 700)
        cy = np.random.randint(100, 700)
        data = TRAIN_DATASET.get_crop(data, cx, cy)
        data = prepare_batch(data, args.data)
        data['spp'] = 1
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)

        optimizer.zero_grad()
        loss_dict = geo_dmtet.tick(glctx, data, mat, loss_fn, i)
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        torch.cuda.empty_cache()

        for k, v in loss_dict.items():
            loss_dict[k] = v.detach().cpu().item()

        if args.wandb:
            wandb.log(loss_dict, step=i)
        else:
            print(f'Iter: {i:05d}', end=' | ')
            for k, v in loss_dict.items():
                print(k, ': ', v, end=' | ')
            print()

        if (i + 1) % 2000 == 0:
            psnr_test = run_validate_crop(glctx, geo_dmtet, mat, VAL_DATASET, os.path.join(logdir, "validate_{}".format(i)), args, device)
            if args.wandb:
                wandb.log({'psnr_test': psnr_test})
            else:
                print(f'psnr_test: {psnr_test}')

    psnr_test = run_validate_crop(glctx, geo_dmtet, mat, VAL_DATASET, os.path.join(logdir, "validate"), args, device)
    if args.wandb:
        wandb.log({'psnr_test': psnr_test})
    else:
        print(f'psnr_test: {psnr_test}')

    pth = os.path.join(logdir, 'geo_dmtet.pth')
    ckpt = {
        'neural_tex_state_dict': mat['neural_tex'].state_dict(),
        'geo_dmtet': geo_dmtet.state_dict(),
    }
    torch.save(ckpt, pth)


if __name__ == '__main__':
    args = load_config()

    if args.test_only:
        test(args)
    else:
        train(args)
