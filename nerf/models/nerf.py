import imageio
import math
import nerfacc
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Callable, Optional


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )


class NerfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        condition_dim: int,  # The number of condition tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
    ):
        super().__init__()
        self.base = MLP(
            input_dim=input_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
        )
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)

        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(
                input_dim=net_width + condition_dim,
                output_dim=3,
                net_depth=net_depth_condition,
                net_width=net_width_condition,
                skip_layer=None,
            )
        else:
            self.rgb_layer = DenseLayer(hidden_features, 3)

    def query_density(self, x):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view(
                    [num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]
                ).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class VanillaNeRFRadianceField(nn.Module):
    def __init__(
        self,
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
    ) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
        )

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x):
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x)
        return F.relu(sigma)

    def forward(self, x, condition=None):
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition)
        return torch.sigmoid(rgb), F.relu(sigma)


class NeRF(nn.Module):

    def __init__(self, args, device, aabb, reco_cur=None):
        super().__init__()
        self.radiance_field = VanillaNeRFRadianceField()
        self.aabb = aabb
        self.device = device
        self.occ_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.aabb.reshape(-1),
            resolution=args.occ_grid_reso,
        )
        self.near = args.near_far[0]
        self.far = args.near_far[1]
        self.render_step_size = args.render_step_size

    def get_optparam_groups(self, optim_conf):
        grad_vars = [{
            'params': self.radiance_field.parameters(),
        }]
        return grad_vars

    def update_step(self, epoch, global_step, args):
        def occ_eval_fn(x):
            density = self.radiance_field.query_density(x)
            return density * self.render_step_size

        # update occupancy grid
        self.occ_grid.update_every_n_steps(
            step=global_step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

    def forward(self, rays, white_bg, is_train=False):
        num_rays = rays.shape[0]
        origins = rays[:, 0:3]
        viewdirs = rays[:, 3:6]

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = self.radiance_field.query_density(positions)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            rgbs, sigmas = self.radiance_field(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends = self.occ_grid.sampling(
            origins,
            viewdirs,
            sigma_fn=sigma_fn,
            near_plane=self.near,
            far_plane=self.far,
            render_step_size=self.render_step_size,
            stratified=is_train,
        )

        rgb, _, depth, _ = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=num_rays,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=white_bg,
        )

        return rgb, depth, t_starts.shape[0]

    def cal_loss(self, data, args):
        rays = data['rays']
        rgbs = data['rgbs']

        rgb_map, _, _ = self.forward(
            rays,
            white_bg=args.white_bg,
            is_train=True,
        )

        loss = F.smooth_l1_loss(rgb_map, rgbs)
        psnr = torch.mean((rgb_map - rgbs) ** 2)
        psnr = -10.0 * torch.log10(psnr)

        return {
            'PSNR': psnr,
            'total_loss': loss,
            'loss': loss,
        }

    def evaluation(
        self,
        dataset,
        args,
        device,
        savePath=None,
        N_vis=-1,
        prefix='',
    ):
        PSNRs = []
        img_eval_interval = 1 if N_vis < 0 else max(dataset.all_rays.shape[0] // N_vis, 1)
        idxs = list(range(0, dataset.all_rays.shape[0], img_eval_interval))
        W, H = dataset.img_wh

        with torch.no_grad():
            self.radiance_field.eval()
            self.occ_grid.eval()
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

            self.radiance_field.train()
            self.occ_grid.train()

        return PSNRs

    def save(self, save_path):
        torch.save(
            {
                "radiance_field_state_dict": self.radiance_field.state_dict(),
                "estimator_state_dict": self.occ_grid.state_dict(),
            },
            save_path,
        )