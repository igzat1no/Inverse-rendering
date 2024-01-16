import imageio
import nerfacc
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml

from models.neus_utils import VolumeSDF, VolumeRadiance
from models.basemodel import BaseModel


class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.variance = nn.Parameter(torch.tensor(config.init_val))

    @property
    def inv_s(self):
        return torch.exp(self.variance * 10.0)

    def forward(self, x):
        return self.inv_s.expand(x.shape[0], 1)


class NeuS(BaseModel):
    def setup(self):
        self.use_rgb_alpha = True

        self.geometry = VolumeSDF(self.config.geometry, self.device)
        self.texture = VolumeRadiance(self.config.texture)
        self.variance = VarianceNetwork(self.config.variance)

        self.background_color = torch.ones([3], dtype=torch.float32, device=self.device)  # white background
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
        self.use_mask_loss = self.config.use_mask

    def occ_eval_fn(self, x):
        sdf, _ = self.geometry(x, with_grad=False)
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(sdf.shape[0], 1)
        estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
        estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
        return alpha

    def update_step(self, epoch, global_step, args):
        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        self.texture.update_step(epoch, global_step)
        self.sampler.update_step(
            global_step,
            self.occ_eval_fn,
            self.config.get('grid_prune_occ_thre', 0.01),
        )

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)
        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def cal_rgb(self, positions, ray_indices, dists):
        t_dirs = self.rays_d[ray_indices]
        sdf, feature, sdf_grad = self.geometry(positions, with_grad=True)
        self.sdf, self.sdf_grad = sdf, sdf_grad
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)
        rgb = self.texture(feature, t_dirs, normal)
        return rgb, alpha

    def cal_loss(self, data, args):
        rays, rgbs_gt, masks_gt = data['rays'], data['rgbs'], data['masks']
        rgb, opacity, depth, _ = self.forward(rays)
        inv_s = self.variance.inv_s
        opacity = opacity.squeeze(-1)
        depth = depth.squeeze(-1)

        loss = 0.
        loss_rgb_mse = F.mse_loss(rgb, rgbs_gt)
        loss += loss_rgb_mse * self.config.loss.lambda_rgb_mse

        loss_rgb_l1 = F.l1_loss(rgb, rgbs_gt)
        loss += loss_rgb_l1 * self.config.loss.lambda_rgb_l1

        loss_eikonal = ((torch.linalg.norm(self.sdf_grad, ord=2, dim=-1) - 1.) ** 2).mean()
        loss += loss_eikonal * self.config.loss.lambda_eikonal

        opacity = torch.clamp(opacity, 1.e-3, 1.-1.e-3)
        loss_mask = F.binary_cross_entropy(opacity, masks_gt.float())
        loss += loss_mask * self.config.loss.lambda_mask if self.use_mask_loss else 0.0

        loss_sparsity = torch.exp(-self.config.loss.sparsity_scale * self.sdf.abs()).mean()
        loss += loss_sparsity * self.config.loss.lambda_sparsity

        return {
            'loss_rgb_mse': loss_rgb_mse,
            'loss_rgb_l1': loss_rgb_l1,
            'loss_eikonal': loss_eikonal,
            'loss_mask': loss_mask,
            'loss_sparsity': loss_sparsity,
            'total_loss': loss,
            'inv_s': inv_s,
            'PSNR': -10.0 * torch.log10(loss_rgb_mse),
        }
