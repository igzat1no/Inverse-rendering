import datetime
from omegaconf import OmegaConf
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from dataset.nerf_synthetic import NerfSyntheticDataset
from dataset.realdata import RealDataset
import models
from utils import *


def load_config():
    register_operation()

    valid_models = ['TensorCP', 'TensorVM', 'TensorVMSplit', 'NeRF', 'NeuS', 'TensoIR']
    valid_datasets = ['nerf_synthetic', 'realdata', 'tensoir_synthetic']
    general_conf = OmegaConf.load('config/general.yaml')
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(general_conf, cli_conf)

    if conf.data.name in valid_datasets:
        data_conf = OmegaConf.load(f'config/{conf.data.name}/{conf.data.type}.yaml')
    else:
        raise NotImplementedError('Unknown dataset type: %s' % conf.data.name)

    if conf.model.name in valid_models:
        model_conf = OmegaConf.load(f'config/model/{conf.model.name}.yaml')
    else:
        raise NotImplementedError('Unknown model: %s' % conf.model.name)

    conf = OmegaConf.merge(general_conf, data_conf, model_conf)

    # ensure all config input from command line are valid
    def search_conf(nw_conf, config):
        for k, v in config.items():
            if k not in nw_conf:
                raise ValueError(f'Unknown config: {k}')
            if hasattr(v, 'keys'):
                search_conf(nw_conf[k], v)
    search_conf(conf, cli_conf)
    conf = OmegaConf.merge(conf, cli_conf)

    return conf


def load_data(data_conf, need_test=False):
    TRAIN_DATASET = None
    VAL_DATASET = None
    TEST_DATASET = None

    print(data_conf)

    if data_conf.name == 'nerf_synthetic' or data_conf.name == 'tensoir_synthetic':
        TRAIN_DATASET = NerfSyntheticDataset(
            datadir=data_conf.dir,
            split='train',
            downsample=data_conf.downsample,
            is_stack=False,
            debug=data_conf.debug,
        )
        VAL_DATASET = NerfSyntheticDataset(
            datadir=data_conf.dir,
            split='val',
            downsample=data_conf.downsample,
            is_stack=True,
            debug=data_conf.debug,
        )
        if need_test:
            TEST_DATASET = NerfSyntheticDataset(
                datadir=data_conf.dir,
                split='test',
                downsample=data_conf.downsample,
                is_stack=True,
                debug=data_conf.debug,
            )
    elif data_conf.name == 'realdata':
        TRAIN_DATASET = RealDataset(
            datadir=data_conf.dir,
            split='train',
            downsample=data_conf.downsample,
            is_stack=False,
        )
        VAL_DATASET = RealDataset(
            datadir=data_conf.dir,
            split='val',
            downsample=data_conf.downsample,
            is_stack=True,
        )
        if need_test:
            TEST_DATASET = RealDataset(
                datadir=data_conf.dir,
                split='test',
                downsample=data_conf.downsample,
                is_stack=True,
            )
    else:
        raise NotImplementedError('Unknown dataset type: %s' % data_conf.name)

    return TRAIN_DATASET, VAL_DATASET, TEST_DATASET


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

    logdir = (f'{args.logdir}/{args.exp}/'
        f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
    print('Logdir: %s' % logdir)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(f'{logdir}/imgs_vis', exist_ok=True)

    # log all config to file
    with open(f'{logdir}/total_config.yaml', 'w') as f:
        OmegaConf.save(args, f)

    # backup source code
    if not args.no_backup:
        backup(logdir, args)

    device = set_device(args.gpu)

    TRAIN_DATASET, VAL_DATASET, TEST_DATASET = load_data(args.data, args.render_test)

    aabb = TRAIN_DATASET.scene_bbox
    args.model.near_far = TRAIN_DATASET.near_far
    args.model.white_bg = getattr(args.data, 'white_bg', TRAIN_DATASET.white_bg)
    grid_size = N_to_reso(getattr(args, 'N_voxel_init', 2097152), aabb)
    nSamples = getattr(args.model, 'nSamples', 1e6)
    stepratio = getattr(args.model, 'step_ratio', 0.5)
    nSamples = min(nSamples, cal_n_samples(grid_size, stepratio))
    print('nSamples: %d' % nSamples)

    model = eval('models.' + args.model.name)(args.model, device, aabb, grid_size).to(device)
    model.nSamples = nSamples
    print('aabb:', model.aabb)
    print('near_far:', model.near_far)

    optim_dict = {
        'Adam': torch.optim.Adam,
    }
    grad_vars = model.get_optparam_groups(args.optimizer)
    optimizer = optim_dict[args.optimizer.name](grad_vars, **args.optimizer.params)
    scheduler = build_scheduler(optimizer, args.scheduler)

    if args.model.name in ['TensorCP', 'TensorVM', 'TensorVMSplit', 'TensoIR']:
        N_voxel_list = (torch.round(torch.exp(torch.linspace(
                            np.log(args.N_voxel_init),
                            np.log(args.N_voxel_final),
                            len(args.upsample.iteration) + 1))).long()).tolist()[1:]

        all_rays, all_rgbs, mask_filtered = filtering_rays(
            model,
            TRAIN_DATASET.all_rays,
            TRAIN_DATASET.all_rgbs,
            device,
            bbox_only=True,
        )
        TRAIN_DATASET.all_rays = all_rays
        TRAIN_DATASET.all_rgbs = all_rgbs

        if args.model.name == 'TensoIR':
            all_light_idx = TRAIN_DATASET.all_light_idx[mask_filtered, :]
            TRAIN_DATASET.all_light_idx = all_light_idx
            args.model.update_AlphaMask_list = args.update_AlphaMask_list
            args.model.iteration = args.iteration

    train_loader = DataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        drop_last=True,
    )
    train_iter = iter(cycle(train_loader))

    print('Number of batches: %d' % len(train_loader))
    PSNRs, PSNRs_test = [], [0]
    PSNRs_brdf, PSNRs_brdf_test = [], [0]

    log_time = False

    for i in tqdm(range(args.iteration)):
        start_time = time.time()

        args.nw_iter = args.model.nw_iter = i
        data = next(train_iter)
        data = {k: v.to(device) for k, v in data.items()}
        if args.model.name == 'TensoIR':
            data['normals'] = None

        model.update_step(epoch=0, global_step=i, args=args)
        loss_dict = model.cal_loss(data, args.model)

        nw_loss_dict = {}
        for k, v in loss_dict.items():
            nw_loss_dict[k] = v.detach().cpu().item()

        total_loss = loss_dict['total_loss']
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        PSNRs.append(nw_loss_dict['PSNR'])
        if args.model.name == 'TensoIR':
            if args.model.relight_flag:
                PSNRs_brdf.append(nw_loss_dict['PSNR_brdf'])

        if args.wandb:
            wandb.log(nw_loss_dict, step=i)
        else:
            print(f'Iter: {i:05d}', end=' | ')
            for k, v in nw_loss_dict.items():
                print(k, ': ', v, end=' | ')
            print()

        if i % 1000 == 0:
            print('step {} model.sampler.aabb {}'.format(i, model.sampler.aabb))

        # eval and vis
        if (i + 1) % args.vis_freq == 0 and args.N_vis != 0:
            model.save(f'{logdir}/{i:06d}')
            PSNRs_test = model.evaluation(
                VAL_DATASET,
                args.model,
                device=device,
                savePath=f'{logdir}/imgs_vis',
                N_vis=args.N_vis,
                prefix=f'{i:06d}_',
            )
            if args.wandb:
                wandb.log({'PSNR_val': np.mean(PSNRs_test)}, step=i)

        if args.model.name in ['TensorCP', 'TensorVM', 'TensorVMSplit', 'TensoIR']:
            if i in args.update_AlphaMask_list:
                if grid_size[0] * grid_size[1] * grid_size[2] < 256 ** 3:
                     reso_mask = grid_size
                new_aabb = model.updateAlphaMask(tuple(reso_mask))
                if i == args.update_AlphaMask_list[0]:
                    model.shrink(new_aabb)
                    # tensorVM.alphaMask = None
                    model.l1_reg_weight = args.model.loss.l1_weight_rest
                    print('continuing L1_reg_weight', model.l1_reg_weight)

                    if args.model.name == 'TensoIR':
                        print("Now adding Relighting")
                        args.model.relight_flag = True
                        torch.cuda.empty_cache()
                        model.tv_weight_density = 0
                        model.tv_weight_app = 0

                if not args.model.ndc_ray and i == args.update_AlphaMask_list[1]:
                    # filter rays outside the bbox
                    all_rays, all_rgbs, mask = filtering_rays(
                        model, all_rays, all_rgbs, device, bbox_only=True)
                    TRAIN_DATASET.all_rays = all_rays
                    TRAIN_DATASET.all_rgbs = all_rgbs
                    if args.model.name == 'TensoIR':
                        all_light_idx = all_light_idx[mask, :]
                        TRAIN_DATASET.all_light_idx = all_light_idx

                    train_loader = DataLoader(
                        TRAIN_DATASET,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=16,
                        drop_last=True,
                    )
                    train_iter = iter(cycle(train_loader))
                    print(len(train_loader))

            if i in args.upsample.iteration:
                n_voxels = N_voxel_list.pop(0)
                grid_size = N_to_reso(n_voxels, model.aabb)
                model.nSamples = min(args.model.nSamples, cal_n_samples(grid_size, stepratio))
                model.upsample_volume_grid(grid_size)

                if args.upsample.lr_reset:
                    print('reset lr to initial')
                    lr_scale = 1
                else:
                    lr_scale = args.lr_decay_target_ratio ** (i / args.iteration)
                grad_vars = model.get_optparam_groups(args.optimizer, lr_scale)
                optimizer = optim_dict[args.optimizer.name](grad_vars, **args.optimizer.params)
                scheduler = build_scheduler(optimizer, args.scheduler)

        if log_time:
            nw_time = time.time()
            print("post process time: ", nw_time - start_time)
            start_time = nw_time

    # save model
    model.save(f'{logdir}')

    if args.render_train and args.model.name not in ['TensorCP', 'TensorVM', 'TensorVMSplit', 'TensoIR']:
        os.makedirs(f'{logdir}/imgs_train_all', exist_ok=True)
        PSNRs_train = model.evaluation(
            TRAIN_DATASET,
            args.model,
            device=device,
            savePath=f'{logdir}/imgs_train_all',
            N_vis=-1,
        )
        wandb.log({'PSNR_train_all': np.mean(PSNRs_train)}, step=args.iteration)
        print(f'======> {args.exp} train all psnr: {np.mean(PSNRs_train)} <========================')

    if args.render_test:
        os.makedirs(f'{logdir}/imgs_test_all', exist_ok=True)
        PSNRs_test = model.evaluation(
            TEST_DATASET,
            args.model,
            device=device,
            savePath=f'{logdir}/imgs_test_all',
            N_vis=-1,
        )
        wandb.log({'PSNR_test_all': np.mean(PSNRs_test)}, step=args.iteration)
        print(f'======> {args.exp} test all psnr: {np.mean(PSNRs_test)} <========================')


if __name__ == '__main__':
    args = load_config()

    if args.test_only:
        raise NotImplementedError
    else:
        train(args)
