import imageio
import nerfacc
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.basemodel import BaseModel
from models.renderer import SHRender, RGBRender, MLPRender, MLPRender_Fea, MLPRender_PE

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device
        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgrid_size = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.grid_size = torch.LongTensor(
            [alpha_volume.shape[-1], alpha_volume.shape[-2],
             alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume,
            xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgrid_size - 1


class TensorBase(BaseModel):
    def setup(self):
        self.density_shift = self.config.density.density_shift
        self.alphaMask_thres = self.config.alpha_mask_thre
        self.distance_scale = self.config.density.distance_scale
        self.step_ratio = self.config.step_ratio
        self.fea2denseAct = self.config.density.fea2dense
        if self.config.white_bg:
            self.background_color = torch.ones(3, device=self.device)
        self.update_render_step_size(self.grid_size)

        self.use_sigma = True
        self.use_rgb_sigma = True

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]
        self.pos_pe, self.view_pe, self.fea_pe = \
            self.config.render.pos_pe, self.config.render.view_pe, self.config.render.fea_pe

        self.init_svd_volume(self.config)
        self.init_render_func(self.config.app.feature_dim, self.config.render)

        self.ortho_reg_weight = self.config.loss.ortho_reg_weight
        self.l1_weight_initial = self.config.loss.l1_weight_initial
        self.l1_weight_rest = self.config.loss.l1_weight_rest
        self.tv_weight_density = self.config.loss.tv_weight_density
        self.tv_weight_app = self.config.loss.tv_weight_app
        self.l1_reg_weight = self.config.loss.l1_weight_initial

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

    def update_render_step_size(self, grid_size):
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.grid_size = torch.LongTensor(grid_size).to(self.device)
        self.units = self.aabbSize / (self.grid_size - 1)
        self.render_step_size = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.render_step_size).item()) + 1

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
        self.update_render_step_size(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l = (xyz_min - self.aabb[0]) / self.units
        b_r = (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(t_l).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)
        self.density.shrink(t_l, b_r)
        self.app.shrink(t_l, b_r)
        if (self.alphaMask is not None) and \
            (not torch.all(self.alphaMask.grid_size == self.grid_size)):
            t_l_r, b_r_r = t_l / (self.grid_size - 1), (b_r - 1) / (self.grid_size - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_render_step_size((newSize[0], newSize[1], newSize[2]))
        print("====> shrinked")

    def density_L1(self):
        return self.density.L1_loss()

    def TV_loss_density(self, reg):
        return self.density.TV_loss(reg)

    def TV_loss_app(self, reg):
        return self.app.TV_loss(reg)

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'grid_size':self.grid_size.tolist(),
            'density_n_comp': self.density.n_comp,
            'app_n_comp': self.app.n_comp,
            'app_dim': self.app.app_dim,
            'near_far': self.near_far,
            'step_ratio': self.step_ratio,
        }

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        print("final density shape is:")
        for i in range(3):
            print(self.density.plane[i].shape)
        ckpt = {'state_dict': self.state_dict(), 'kwargs': self.get_kwargs()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path + '/model.pt')

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
            alpha_mask = torch.ones_like(xyz_locs[:, 0]).bool()

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature_with_grad(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])
        return alpha

    @torch.no_grad()
    def getDenseAlpha(self, grid_size=None):
        grid_size = self.grid_size.tolist() if grid_size is None else grid_size

        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, grid_size[0]),
                torch.linspace(0, 1, grid_size[1]),
                torch.linspace(0, 1, grid_size[2]),
            ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples
        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(grid_size[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1, 3), self.render_step_size).view(
                (grid_size[1], grid_size[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, grid_size=(200, 200, 200)):
        alpha, dense_xyz = self.getDenseAlpha(grid_size)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = grid_size[0] * grid_size[1] * grid_size[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2,
                             stride=1).view(grid_size[::-1])
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
        else:
            raise NotImplementedError

    def compute_density(self, x):
        return (self.feature2density(self.compute_densityfeature(x)) *
                self.distance_scale)

    def update_step(self, epoch, global_step, args):
        def occ_normalize_coord(xyz_sampled):
            return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1
        def occ_eval_fn(x):
            step_size = self.render_step_size
            density = self.compute_density(occ_normalize_coord(x))[:, None]
            return density * step_size

        self.sampler.update_step(global_step, occ_eval_fn)

        self.tv_weight_density *= args.lr_factor
        self.tv_weight_app *= args.lr_factor

    def cal_sigma(self, positions, ray_indices):
        return self.compute_density(self.normalize_coord(positions))

    def cal_rgb(self, positions, ray_indices, dists):
        t_dirs = self.rays_d[ray_indices]
        positions = self.normalize_coord(positions)
        sigmas = self.compute_density(positions)
        rgbs = self.renderModule(positions, t_dirs, self.compute_appfeature(positions))
        return rgbs, sigmas

    def cal_loss(self, data, args):
        rays = data['rays']
        rgb_gt = data['rgbs']

        N_rays = rays.shape[0]
        nw_rgb, _, _, _ = self.forward(rays, is_train=True)

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