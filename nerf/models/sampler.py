import nerfacc
import torch
import torch.nn as nn


class Base_sampler(nn.Module):
    def __init__(self, config, aabb):
        super().__init__()

        self.config = config
        self.aabb = aabb

    def sample(self):
        raise NotImplementedError

    def update_step(self, global_step, occ_eval_fn, occ_thre=0.01):
        return


class Vanilla_Sampler(Base_sampler):
    def __init__(self, config, aabb):
        super(Vanilla_Sampler, self).__init__(config, aabb)

    def sample(
        self,
        rays_o,
        rays_d,
        near_far,
        step_size,
        N_samples,
        alphaMask=None,
        is_train=False,
    ):
        near, far = near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples + 1)[None].float() # [1, N_samples + 1]
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = step_size * rng.to(rays_o.device)
        interpx = (t_min[..., None] + step)
        t_start = interpx[..., :-1]
        t_end = interpx[..., 1:]
        t_pos = (t_start + t_end) / 2
        indices = torch.arange(rays_d.shape[-2]).to(rays_o.device)
        indices = indices[..., None].repeat(1, N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * t_pos[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        mask_outbbox = ~mask_outbbox

        if alphaMask is not None:
            alphas = alphaMask.sample_alpha(rays_pts[mask_outbbox])
            alpha_mask = alphas > 0
            tmp_mask = mask_outbbox.clone()
            mask_outbbox[tmp_mask] = alpha_mask

        indices = indices[mask_outbbox]
        t_start = t_start[mask_outbbox]
        t_end = t_end[mask_outbbox]

        return indices, t_start, t_end


class Occgrid_sampler(Base_sampler):
    def __init__(self, config, aabb):
        super(Occgrid_sampler, self).__init__(config, aabb)

        self.occ_grid = nerfacc.OccGridEstimator(
            roi_aabb=aabb.reshape(-1),
            resolution=config.occ_grid_reso,
        )

    def sample(
        self,
        rays_o,
        rays_d,
        sigma_fn,
        alpha_fn,
        near_far,
        render_step_size,
        is_train=False,
    ):
        return self.occ_grid.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            alpha_fn=alpha_fn,
            near_plane=near_far[0],
            far_plane=near_far[1],
            render_step_size=render_step_size,
            stratified=is_train,
        )

    def update_step(self, global_step, occ_eval_fn, occ_thre=0.01):
        self.occ_grid.update_every_n_steps(
            step=global_step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=occ_thre,
        )
