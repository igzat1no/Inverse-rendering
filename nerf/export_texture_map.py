import json
import os
import sys

import imageio.v2 as imageio
import nvdiffrast.torch as nvdr
import drjit as dr
import mitsuba as mi
import models

import numpy as np
import trimesh
import xatlas
import cv2
import open3d as o3d
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion
from omegaconf import OmegaConf
from utils import *
from render import light, material, mlptexture, util

mi.set_variant('cuda_ad_rgb')


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


def contract(xyzs):
    if isinstance(xyzs, np.ndarray):
        mag = np.max(np.abs(xyzs), axis=1, keepdims=True)
        xyzs = np.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
    else:
        mag = torch.amax(torch.abs(xyzs), dim=1, keepdim=True)
        xyzs = torch.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
    return xyzs


def uncontract(xyzs):
    if isinstance(xyzs, np.ndarray):
        mag = np.max(np.abs(xyzs), axis=1, keepdims=True)
        xyzs = np.where(mag <= 1, xyzs, xyzs * (1 / (2 * mag - mag * mag)))
    else:
        mag = torch.amax(torch.abs(xyzs), dim=1, keepdim=True)
        xyzs = torch.where(mag <= 1, xyzs, xyzs * (1 / (2 * mag - mag * mag)))
    return xyzs


if __name__ == '__main__':
    assert(len(sys.argv) == 3)

    stage2_path = sys.argv[-2]
    nw_path = os.path.join('logs', sys.argv[-1])
    os.makedirs(nw_path, exist_ok=True)
    model_path = os.path.join(stage2_path, 'geo_dmtet.pth')

    device = set_device(0)
    model = torch.load(model_path, map_location=device)
    geo_dmtet = model['geo_dmtet']
    mat = model['neural_tex_state_dict']

    exit(0)

    args = load_config()
    seed_everything(args.seed)
    torch.set_default_dtype(torch.float32)
    logdir = './stage3debug'
    os.makedirs(logdir, exist_ok=True)
    device = set_device(args.gpu)

    args.model.geo_model.near_far = [2.0, 6.0]
    args.model.app_model.near_far = [2.0, 6.0]
    args.model.geo_model.white_bg = True
    args.model.app_model.white_bg = True

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

    app_model.is_relight = True

    geo_model_dmtet_name = args.model.geo_model.name + '_DMTet'
    geo_dmtet = eval('models.' + geo_model_dmtet_name)(geo_model, device, args.model).to(device)

    mat = initial_guess_material(geo_dmtet, args.model, tensorf_model=app_model)
    glctx = nvdr.RasterizeCudaContext()

    stage2_path = './logs/hotdog-stage2/2023_08_18_23_42_20/validate'
    hotdog_path = stage2_path + '/mesh.ply'
    envmap_path = stage2_path + '/envirmap.png'
    model_path = stage2_path + '/../geo_dmtet.pth'
    data_path = './data/tensoir/hotdog'
    with open(os.path.join(data_path, 'transforms_train.json'), 'r') as f:
        meta = json.load(f)

    ckpt = torch.load(model_path, map_location=device)
    geo_dmtet.load_state_dict(ckpt['geo_dmtet'])
    mat['neural_tex'].load_state_dict(ckpt['neural_tex_state_dict'])

    mesh = trimesh.load_mesh("/home/zongtai/nerf/logs/hotdog-stage2/2023_08_18_23_42_20/validate/mesh.ply")

    # print("start")
    # # start time
    # start = time.time()
    # vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    # xatlas.export("output.obj", mesh.vertices[vmapping], indices, uvs)
    # print("finished")
    # print('Time: ', time.time() - start)

    v = mesh.vertices
    f = mesh.faces

    print(type(v))

    def _export_obj(v, f, h0, w0, ssaa=1, cas=0, path=''):
        v = torch.tensor(v, dtype=torch.float32, device=torch.device('cuda'))
        f = torch.tensor(f, dtype=torch.int32, device=torch.device('cuda'))

        # v, f: torch Tensor

        v_np = v.cpu().numpy() # [N, 3]
        f_np = f.cpu().numpy() # [M, 3]
        # v_np = np.array(v, dtype=np.float32) # [N, 3]
        # print(v_np.max(), v_np.min())
        # f_np = np.array(f, dtype=np.int32) # [M, 3]

        print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

        # unwrap uv in contracted space
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        print('finished adding mesh')
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 0 # disable merge_chart for faster unwrap...
        pack_options = xatlas.PackOptions()
        # pack_options.blockAlign = True
        # pack_options.bruteForce = False
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        print('finished generation')
        vmapping, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]

        # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

        vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

        # render uv maps
        uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

        if ssaa > 1:
            h = int(h0 * ssaa)
            w = int(w0 * ssaa)
        else:
            h, w = h0, w0

        glctx = nvdr.RasterizeCudaContext()

        rast, _ = nvdr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
        xyzs, _ = nvdr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
        mask, _ = nvdr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]

        print("done here")

        # masked query
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)

        print(mask.shape, mask.sum())

        # if self.opt.contract:
        #     xyzs = contract(xyzs)

        final_albedo = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
        final_roughness = torch.zeros(h * w, 1, device=device, dtype=torch.float32)
        final_normal = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
        print(h, w)
        print(xyzs.shape)

        # save the xyzs
        nw_points = xyzs.reshape(-1, 3)
        nw_points = nw_points.detach().cpu().numpy()
        print(nw_points.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(nw_points)
        o3d.io.write_point_cloud('./stage3debug/pts.ply', pcd)

        print(xyzs.max(), xyzs.min(), 'before')
        aabbsize = mat['neural_tex'].net.aabb[1] - mat['neural_tex'].net.aabb[0]
        xyzs = (xyzs - mat['neural_tex'].net.aabb[0]) * 1.0 / aabbsize * 2 - 1
        print(xyzs.max(), xyzs.min())

        if mask.any():
            xyzs = xyzs[mask] # [M, 3]
            print(xyzs.shape)
            nw_points = xyzs.reshape(-1, 3)
            nw_points = nw_points.detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(nw_points)
            o3d.io.write_point_cloud('./stage3debug/after_mask_pts.ply', pcd)

            # check individual codes
            ind_code = None

            # batched inference to avoid OOM
            all_albedo = []
            all_roughness = []
            all_normal = []
            head = 0
            while head < xyzs.shape[0]:
                tail = min(head + 640000, xyzs.shape[0])
                print('finished 1 ci')
                with torch.cuda.amp.autocast(enabled=False):
                    positions = xyzs
                    intrinsic_feat = mat['neural_tex'].net.compute_intrinfeature(positions)
                    brdf = mat['neural_tex'].net.renderModule_brdf(positions, intrinsic_feat)
                    albedo, roughness = brdf[..., :3], (brdf[..., 3:4] * 0.9 + 0.09)
                    normal = mat['neural_tex'].net.renderModule_normal(positions, intrinsic_feat)
                    all_albedo.append(albedo)
                    all_roughness.append(roughness)
                    all_normal.append(normal)

                head += 640000

            final_albedo[mask] = torch.cat(all_albedo, dim=0)
            final_roughness[mask] = torch.cat(all_roughness, dim=0)
            final_normal[mask] = torch.cat(all_normal, dim=0)

        final_albedo = final_albedo.view(h, w, -1) # 6 channels
        final_roughness = final_roughness.view(h, w, -1) # 6 channels
        final_normal = final_normal.view(h, w, -1)
        mask = mask.view(h, w)
        print(final_albedo.max(), final_albedo.min())
        print(final_roughness.max(), final_roughness.min())

        # quantize [0.0, 1.0] to [0, 255]
        final_albedo = final_albedo.detach().cpu().numpy()
        final_albedo = (final_albedo * 255).astype(np.uint8)

        final_roughness = final_roughness.repeat(1, 1, 3)
        final_roughness = final_roughness.detach().cpu().numpy()
        final_roughness = (final_roughness * 255).astype(np.uint8)

        final_normal = F.normalize(final_normal, p=2, dim=-1, eps=1e-6)
        final_normal = final_normal.detach().cpu().numpy()
        final_normal = (final_normal * 255).astype(np.uint8)

        ### NN search as a queer antialiasing ...
        mask = mask.cpu().numpy()

        inpaint_region = binary_dilation(mask, iterations=32) # pad width
        inpaint_region[mask] = 0

        search_region = mask.copy()
        not_search_region = binary_erosion(search_region, iterations=3)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
        _, indices = knn.kneighbors(inpaint_coords)

        final_albedo[tuple(inpaint_coords.T)] = final_albedo[tuple(search_coords[indices[:, 0]].T)]

        final_roughness[tuple(inpaint_coords.T)] = final_roughness[tuple(search_coords[indices[:, 0]].T)]

        final_normal[tuple(inpaint_coords.T)] = final_normal[tuple(search_coords[indices[:, 0]].T)]


        # do ssaa after the NN search, in numpy
        feats0 = cv2.cvtColor(final_albedo[..., :3], cv2.COLOR_RGB2BGR) # albedo
        feats1 = cv2.cvtColor(final_roughness[..., :3], cv2.COLOR_RGB2BGR) # roughness
        feats2 = cv2.cvtColor(final_normal[..., :3], cv2.COLOR_RGB2BGR)

        if ssaa > 1:
            feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
            feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)
            feats2 = cv2.resize(feats2, (w0, h0), interpolation=cv2.INTER_LINEAR)

        # cv2.imwrite(os.path.join(path, f'feat0_{cas}.png'), feats0, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
        # cv2.imwrite(os.path.join(path, f'feat1_{cas}.png'), feats1, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
        cv2.imwrite(os.path.join(path, f'feat0_{cas}.jpg'), feats0)
        cv2.imwrite(os.path.join(path, f'feat1_{cas}.jpg'), feats1)
        cv2.imwrite(os.path.join(path, f'normal_{cas}.jpg'), feats2)

        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f'mesh_{cas}.obj')
        mtl_file = os.path.join(path, f'mesh_{cas}.mtl')

        print(f'[INFO] writing obj mesh to {obj_file}')
        with open(obj_file, "w") as fp:

            fp.write(f'mtllib mesh_{cas}.mtl \n')

            print(f'[INFO] writing vertices {v_np.shape}')
            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            print(f'[INFO] writing vertices texture coords {vt_np.shape}')
            for v in vt_np:
                fp.write(f'vt {v[0]} {1 - v[1]} \n')

            print(f'[INFO] writing faces {f_np.shape}')
            fp.write(f'usemtl defaultMat \n')
            for i in range(len(f_np)):
                fp.write(f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl defaultMat \n')
            fp.write(f'Ka 1 1 1 \n')
            fp.write(f'Kd 1 1 1 \n')
            fp.write(f'Ks 0 0 0 \n')
            fp.write(f'Tr 1 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0 \n')
            fp.write(f'map_Kd normal_{cas}.jpg \n')

    resolution = 512
    _export_obj(v, f, resolution, resolution, ssaa=2, cas=0, path='./stage3debug')