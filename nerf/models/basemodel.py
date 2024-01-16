import os
import sys

import imageio
import nerfacc
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from models import sampler


class BaseModel(nn.Module):
    def __init__(self, config, device, aabb, grid_size, shit=False):
        super().__init__()

        self.config = config
        self.device = device
        self.aabb = aabb.to(device)
        self.grid_size = grid_size
        self.near_far = getattr(config, 'near_far', [0., 100000.])
        self.render_step_size = 0.
        self.use_rgb_sigma = False
        self.use_rgb_alpha = False
        self.use_sigma = False
        self.use_alpha = False
        self.background_color = None
        self.alphaMask = None
        if config.sampler == 'occgrid':
            self.sampler = sampler.Occgrid_sampler(config, self.aabb)
        elif config.sampler == 'vanilla':
            self.sampler = sampler.Vanilla_Sampler(config, self.aabb)
        else:
            raise NotImplementedError(f'No such sampler: {config.sampler}')

        self.shit=shit

        self.setup()

    def setup(self):
        raise NotImplementedError('Please implement setup() method')

    def get_optparam_groups(self, args):
        return self.parameters()

    def update_step(self, epoch, global_step, args):
        raise NotImplementedError('Please implement update_step() method')

    def get_positions(self, t_starts, t_ends, ray_indices):
        t_origins = self.rays_o[ray_indices]
        t_dirs = self.rays_d[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        return positions

    def cal_sigma(self, positions, ray_indices):
        raise NotImplementedError('Please implement cal_sigma() method')

    def cal_alpha(self, positions, ray_indices):
        raise NotImplementedError('Please implement cal_alpha() method')

    def cal_rgb(self, positions, ray_indices, dists):
        raise NotImplementedError('Please implement cal_rgb() method')

    def rgb_fn(self, t_starts, t_ends, ray_indices):
        positions = self.get_positions(t_starts, t_ends, ray_indices)
        if positions.shape[0] == 0:
            return torch.zeros((0, 3), device=self.device), \
                torch.zeros((0,), device=self.device)
        return self.cal_rgb(positions, ray_indices, t_ends - t_starts)

    def sigma_fn(self, t_starts, t_ends, ray_indices):
        positions = self.get_positions(t_starts, t_ends, ray_indices)
        if positions.shape[0] == 0:
            return torch.zeros((0,), device=self.device)
        return self.cal_sigma(positions, ray_indices)

    def alpha_fn(self, t_starts, t_ends, ray_indices):
        positions = self.get_positions(t_starts, t_ends, ray_indices)
        if positions.shape[0] == 0:
            return torch.zeros((0,), device=self.device)
        return self.cal_alpha(positions, ray_indices)

    def forward(self, rays, is_train=False):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, :3], rays[:, 3:6]
        self.rays_o, self.rays_d = rays_o, rays_d

        if self.config.sampler == 'occgrid':
            ray_indices, t_starts, t_ends = self.sampler.sample(
                rays_o,
                rays_d,
                sigma_fn=self.sigma_fn if self.use_sigma else None,
                alpha_fn=self.alpha_fn if self.use_alpha else None,
                near_far=self.near_far,
                render_step_size=self.render_step_size,
                is_train=is_train,
            )
        elif self.config.sampler == 'vanilla':
            ray_indices, t_starts, t_ends = self.sampler.sample(
                rays_o,
                rays_d,
                self.near_far,
                self.render_step_size,
                N_samples=self.nSamples,
                alphaMask=self.alphaMask,
                is_train=is_train,
            )

        rgb, opacity, depth, extras = nerfacc.rendering(
            t_starts, t_ends,
            ray_indices=ray_indices,
            n_rays=n_rays,
            rgb_sigma_fn=self.rgb_fn if self.use_rgb_sigma else None,
            rgb_alpha_fn=self.rgb_fn if self.use_rgb_alpha else None,
            render_bkgd=self.background_color,
        )

        return rgb, opacity, depth, extras

    def cal_loss(self, data, args):
        pass

    def evaluation(
        self,
        dataset,
        args,
        device,
        savePath,
        N_vis=-1,
        prefix='',
    ):
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
                    nw_rgb, _, nw_depth, _ = self.forward(
                        nw_rays,
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

    def save(self, dir) -> None:
        # save config
        os.makedirs(dir, exist_ok=True)
        OmegaConf.save(self.config, os.path.join(dir, 'config.yaml'))
        torch.save({
            'state_dict': self.state_dict(),
            'aabb': self.aabb,
            'grid_size': self.grid_size,
        }, os.path.join(dir, 'model.pth'))
