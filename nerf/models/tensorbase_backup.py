import imageio
import nerfacc
from nerfacc import accumulate_along_rays
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.renderer import SHRender, RGBRender, MLPRender, MLPRender_Fea, MLPRender_PE
from models.tensoIR.relight_utils import grid_sample

class CPModule(nn.Module):
    '''Factorize the model with CP decomposition.
    '''
    def __init__(self, n_comp, gridSize, scale=0.2, dim=3):
        super(CPModule, self).__init__()
        self.n_comp = n_comp
        self.gridSize = gridSize
        self.scale = scale
        self.dim = dim

        self.param = []
        for i in range(dim):
            self.param.append(
                nn.Parameter(scale * torch.randn(1, n_comp, gridSize[i], 1)))
        self.param = nn.ParameterList(self.param)

    def compute_feature(self, xyz_sampled):
        coordinate_line = torch.stack(
            [xyz_sampled[:, i] for i in range(self.dim)])
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line),
            dim=-1).detach().view(3, -1, 1, 2)

        line_coef_point = F.grid_sample(
            self.param[0], coordinate_line[[0]], align_corners=True).view(
                -1, *xyz_sampled.shape[:1])
        for i in range(1, self.dim):
            line_coef_point = line_coef_point * F.grid_sample(
                self.param[i], coordinate_line[[i]], align_corners=True).view(
                    -1, *xyz_sampled.shape[:1])

        return line_coef_point

    @torch.no_grad()
    def upsample(self, res_target):
        for i in range(self.dim):
            self.param[i] = nn.Parameter(
                F.interpolate(self.param[i].data,
                              size=(res_target[i], 1),
                              mode="bilinear",
                              align_corners=True))

    @torch.no_grad()
    def shrink(self, l, r):
        for i in range(self.dim):
            self.param[i] = nn.Parameter(
                self.param[i].data[:, :, l[i]:r[i], :])

    def L1_loss(self):
        loss = 0
        for i in range(self.dim):
            loss += torch.mean(torch.abs(self.param[i]))
        return loss

    def TV_loss(self, reg):
        loss = 0
        for i in range(self.dim):
            loss += reg(self.param[i]) * 1e-3
        return loss


class DensityLine(CPModule):
    def __init__(self, n_comp, gridSize, scale=0.2, dim=3):
        super(DensityLine, self).__init__(n_comp, gridSize, scale, dim)

    def compute(self, xyz_sampled):
        line_coef_point = self.compute_feature(xyz_sampled)
        return torch.sum(line_coef_point, dim=0)


class AppLine(CPModule):
    def __init__(self, n_comp, gridSize, app_dim, scale=0.2, dim=3):
        super(AppLine, self).__init__(n_comp, gridSize, scale, dim)
        self.app_dim = app_dim
        self.mat = nn.Linear(n_comp, app_dim, bias=False)

    def compute(self, xyz_sampled):
        line_coef_point = self.compute_feature(xyz_sampled)
        return self.mat(line_coef_point.T)


class VMModule(nn.Module):
    '''Plane + line, for VM decomposition.
    '''
    def __init__(self, n_comp, gridSize, scale=0.1, dim=3):
        super(VMModule, self).__init__()
        self.n_comp = n_comp
        self.gridSize = gridSize
        self.scale = scale
        self.dim = dim

        self.matMode = []
        for i in range(dim):
            nw = []
            for j in range(dim):
                if j != i:
                    nw.append(j)
            self.matMode.append(nw)

        self.plane, self.line = [], []
        # now we only implement for dim=3
        for i in range(self.dim):
            id_0, id_1 = self.matMode[i]
            self.plane.append(
                nn.Parameter(scale * torch.randn(1, n_comp[i], gridSize[id_1], gridSize[id_0])))
            self.line.append(
                nn.Parameter(scale * torch.randn(1, n_comp[i], gridSize[i], 1)))
        self.plane = nn.ParameterList(self.plane)
        self.line = nn.ParameterList(self.line)

    def compute_feature(self, xyz_sampled):
        coordinate_plane = torch.stack([xyz_sampled[..., self.matMode[i]]
            for i in range(self.dim)]).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack([xyz_sampled[:, i] for i in range(self.dim)])
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line),
            dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for i in range(self.dim):
            plane_coef_point.append(F.grid_sample(self.plane[i], coordinate_plane[[i]],
                        align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.line[i], coordinate_line[[i]],
                        align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point = torch.cat(plane_coef_point)
        line_coef_point = torch.cat(line_coef_point)

        return plane_coef_point * line_coef_point

    def compute_feature_with_grad(self, xyz_sampled):
        coordinate_plane = torch.stack([xyz_sampled[..., self.matMode[i]]
            for i in range(self.dim)]).view(3, -1, 1, 2)
        coordinate_line = torch.stack([xyz_sampled[:, i] for i in range(self.dim)])
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line),
            dim=-1).view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for i in range(self.dim):
            plane_coef_point.append(grid_sample(self.plane[i],
                coordinate_plane[[i]]).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(grid_sample(self.line[i],
                coordinate_line[[i]]).view(-1, *xyz_sampled.shape[:1]))

        plane_coef_point = torch.cat(plane_coef_point)
        line_coef_point = torch.cat(line_coef_point)
        return plane_coef_point * line_coef_point

    def vectorDiff(self):
        total = 0
        for i in range(self.dim):
            n_comp, n_size = self.line[i].shape[1:-1]
            dotp = torch.matmul(self.line[i].view(n_comp, n_size),
                                self.line[i].view(n_comp, n_size).transpose(-1, -2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total += torch.mean(torch.abs(non_diagonal))
        return total

    @torch.no_grad()
    def upsample(self, res_target):
        for i in range(self.dim):
            id_0, id_1 = self.matMode[i]
            self.plane[i] = nn.Parameter(
                F.interpolate(self.plane[i].data,
                              size=(res_target[id_1], res_target[id_0]),
                              mode='bilinear',
                              align_corners=True))
            self.line[i] = nn.Parameter(
                F.interpolate(self.line[i].data,
                              size=(res_target[i], 1),
                              mode='bilinear',
                              align_corners=True))

    @torch.no_grad()
    def shrink(self, t_l, b_r):
        for i in range(self.dim):
            self.line[i] = nn.Parameter(
                self.line[i].data[..., t_l[i]:b_r[i], :])
            id_0, id_1 = self.matMode[i]
            self.plane[i] = nn.Parameter(
                self.plane[i].data[..., t_l[id_1]:b_r[id_1], t_l[id_0]:b_r[id_0]])

    def L1_loss(self):
        loss = 0
        for i in range(self.dim):
            loss += torch.mean(torch.abs(self.plane[i]))
            loss += torch.mean(torch.abs(self.line[i]))
        return loss

    def TV_loss(self, reg):
        loss = 0
        for i in range(self.dim):
            loss += reg(self.plane[i]) * 1e-2  # + reg(self.line[i]) * 1e-3
        return loss


class DensityVM(VMModule):
    def __init__(self, n_comp, gridSize, scale=0.1, dim=3):
        super(DensityVM, self).__init__(n_comp, gridSize, scale, dim)

    def compute(self, xyz_sampled):
        feat = self.compute_feature(xyz_sampled)
        return torch.sum(feat, dim=0)

    def compute_with_grad(self, xyz_sampled):
        feat = self.compute_feature_with_grad(xyz_sampled)
        return torch.sum(feat, dim=0)


class AppVM(VMModule):
    def __init__(self, n_comp, gridSize, app_dim, scale=0.1, dim=3):
        super(AppVM, self).__init__(n_comp, gridSize, scale, dim)
        self.app_dim = app_dim
        self.mat = nn.Linear(sum(n_comp), app_dim, bias=False)

    def compute(self, xyz_sampled):
        feat = self.compute_feature(xyz_sampled)
        return self.mat(feat.T)

    def compute_with_grad(self, xyz_sampled):
        feat = self.compute_feature_with_grad(xyz_sampled)
        return self.mat(feat.T)


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device
        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor(
            [alpha_volume.shape[-1], alpha_volume.shape[-2],
             alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume,
            xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1


class TensorBase(nn.Module):

    def __init__(self, args, device, aabb, gridSize):
        super(TensorBase, self).__init__()

        self.alphaMask = None
        self.near_far = args.near_far
        self.density_shift = args.density.density_shift
        self.alphaMask_thres = args.alpha_mask_thre
        self.distance_scale = args.density.distance_scale
        self.step_ratio = args.step_ratio
        self.fea2denseAct = args.density.fea2dense
        self.aabb = aabb.to(device)
        self.occ_grid = nerfacc.OccGridEstimator(
            roi_aabb=aabb.reshape(-1),
            resolution=args.occ_grid_reso,
        )
        self.white_bg = args.white_bg
        self.ndc_ray = args.ndc_ray
        self.device = device

        self.update_stepSize(gridSize)

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]
        self.pos_pe, self.view_pe, self.fea_pe = \
            args.render.pos_pe, args.render.view_pe, args.render.fea_pe

        self.init_svd_volume(args)
        self.init_render_func(args.app.feature_dim, args.render)

        self.ortho_reg_weight = args.loss.ortho_reg_weight
        self.l1_weight_initial = args.loss.l1_weight_initial
        self.l1_weight_rest = args.loss.l1_weight_rest
        self.tv_weight_density = args.loss.tv_weight_density
        self.tv_weight_app = args.loss.tv_weight_app
        self.l1_reg_weight = args.loss.l1_weight_initial

    def init_render_func(self, app_dim, conf):
        if conf.name == 'MLP_PE':
            self.renderModule = MLPRender_PE(app_dim,
                conf.view_pe, conf.pos_pe, conf.featureC)
        elif conf.name == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(app_dim,
                conf.view_pe, conf.fea_pe, conf.featureC)
        elif conf.name == 'MLP':
            self.renderModule = MLPRender(app_dim, conf.view_pe, conf.featureC)
        elif conf.name == 'SH':
            self.renderModule = SHRender
        elif conf.name == 'RGB':
            assert app_dim == 3
            self.renderModule = RGBRender
        else:
            raise NotImplementedError('Unknown shading mode: %s' % conf.name)
        print("pos_pe", conf.pos_pe, "view_pe", conf.view_pe, "fea_pe", conf.fea_pe)
        print("renderModule", self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))  # tensor([-1.5000, -1.5000, -1.5000,  1.5000,  1.5000,  1.5000])
        print("grid size", gridSize)  # [128, 128, 128]
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, args):
        pass

    def compute_densityfeature(self, xyz_sampled):
        return self.density.compute(xyz_sampled)

    def compute_densityfeature_with_grad(self, xyz_sampled):
        return self.density.compute_with_grad(xyz_sampled)

    def compute_appfeature(self, xyz_sampled):
        return self.app.compute(xyz_sampled)

    def compute_appfeature_with_grad(self, xyz_sampled):
        return self.app.compute_with_grad(xyz_sampled)

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density.upsample(res_target)
        self.app.upsample(res_target)
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l = (xyz_min - self.aabb[0]) / self.units
        b_r = (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(t_l).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)
        self.density.shrink(t_l, b_r)
        self.app.shrink(t_l, b_r)
        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
        print("====> shrinked")

    def density_L1(self):
        return self.density.L1_loss()

    def TV_loss_density(self, reg):
        return self.density.TV_loss(reg)

    def TV_loss_app(self, reg):
        return self.app.TV_loss(reg)

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density.n_comp,
            'app_n_comp': self.app.n_comp,
            'app_dim': self.app.app_dim,
            'near_far': self.near_far,
            'step_ratio': self.step_ratio,
        }

    def save(self, path):
        for i in range(3):
            print(self.density.plane[i].shape)
        ckpt = {'state_dict': self.state_dict(), 'kwargs': self.get_kwargs()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(
                    ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device,
                ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])

    def compute_alpha(self, xyz_locs, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])
        return alpha

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize.tolist() if gridSize is None else gridSize

        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
            ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples
        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1, 3), self.stepSize).view(
                (gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200)):
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2,
                             stride=1).view(gridSize[::-1])
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100))
        return new_aabb

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_density(self, x):
        return (self.feature2density(self.compute_densityfeature(x)) *
                self.distance_scale)

    def update_step(self, epoch, global_step, args):
        def occ_normalize_coord(xyz_sampled):
            return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1
        def occ_eval_fn(x):
            step_size = self.stepSize
            density = self.compute_density(occ_normalize_coord(x))[:, None]
            return density * step_size

        self.occ_grid.update_every_n_steps(
            step=global_step, occ_eval_fn=occ_eval_fn
        )

        self.tv_weight_density *= args.lr_factor
        self.tv_weight_app *= args.lr_factor

    def forward(
        self,
        rays_chunk,
        white_bg=True,
        is_train=False,
        ndc_ray=False,
        N_samples=-1,
    ):
        assert not ndc_ray
        origins = rays_chunk[:, :3]
        viewdirs = rays_chunk[:, 3:6]

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            if t_origins.shape[0] == 0:
                return torch.zeros((0,), device=t_origins.device)
            return self.compute_density(self.normalize_coord(positions))

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = viewdirs[ray_indices]
            if t_origins.shape[0] == 0:
                return torch.zeros(
                    (0, 3), device=t_origins.device
                ), torch.zeros((0,), device=t_origins.device)
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            positions = self.normalize_coord(positions)
            sigmas = self.compute_density(positions)
            rgbs = self.renderModule(
                positions, t_dirs, self.compute_appfeature(positions)
            )
            # print('rgbs', rgbs.shape)
            return rgbs, sigmas

        ray_indices, t_starts, t_ends = self.occ_grid.sampling(
            origins,
            viewdirs,
            sigma_fn=sigma_fn,
            near_plane=self.near_far[0],
            far_plane=self.near_far[1],
            render_step_size=self.stepSize,
            stratified=is_train,
        )
        rgb_map, _, depth_map, _ = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices=ray_indices,
            n_rays=origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=1 if white_bg else 0,
        )

        return rgb_map, depth_map, t_starts.shape[0]

    def cal_loss(self, data, args):
        rays = data['rays']
        rgb_gt = data['rgbs']

        N_rays = rays.shape[0]
        nw_rgb, _, _ = self.forward(
            rays,
            white_bg=self.white_bg,
            is_train=True,
            ndc_ray=self.ndc_ray,
            N_samples=4096,
        )

        loss = torch.mean((nw_rgb - rgb_gt) ** 2)
        total_loss = loss
        loss_reg_l1 = self.density_L1()
        if self.l1_reg_weight > 0:
            total_loss += self.l1_reg_weight * loss_reg_l1

        return {
            'total_loss': total_loss,
            'loss': loss,
            'l1_loss': loss_reg_l1,
            'PSNR': -10.0 * torch.log10(loss),
        }

    def evaluation(
        self,
        dataset,
        args,
        device,
        savePath=None,
        N_vis=-1,
        prefix='',
        white_bg=False):
        PSNRs = []
        img_eval_interval = 1 if N_vis < 0 else max(dataset.all_rays.shape[0] // N_vis, 1)
        idxs = list(range(0, dataset.all_rays.shape[0], img_eval_interval))
        W, H = dataset.img_wh

        with torch.no_grad():
            self.eval()
            for idx, samples in tqdm(enumerate(dataset.all_rays[0::img_eval_interval]),
                                     file=sys.stdout):
                rays = samples.view(-1,samples.shape[-1])

                chunk = 4096
                N_rays = rays.shape[0]
                rgb = []
                depth = []
                for i in range(N_rays // chunk + int(N_rays % chunk > 0)):
                    nw_rays = rays[i * chunk:(i + 1) * chunk].to(device)
                    nw_rgb, nw_depth, _ = self.forward(
                        nw_rays,
                        white_bg=args.model.white_bg,
                        is_train=False,
                    )
                    rgb.append(nw_rgb.cpu())
                    depth.append(nw_depth.cpu())
                rgb = torch.cat(rgb)
                depth = torch.cat(depth)

                rgb = rgb.clamp(0.0, 1.0)
                rgb, depth = rgb.reshape(H, W, 3), depth.reshape(H, W)
                gt_rgb = dataset.all_rgbs[idxs[idx]].view(H, W, 3)
                loss = torch.mean((rgb - gt_rgb) ** 2)
                PSNRs.append(-10.0 * torch.log10(loss))
                rgb = (rgb.numpy() * 255).astype('uint8')
                if savePath is not None:
                    imageio.imwrite(f'{savePath}/{prefix}{idx:03d}.png', rgb)

            self.train()

        return PSNRs