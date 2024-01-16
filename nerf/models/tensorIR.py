from nerfacc import accumulate_along_rays
from tqdm import tqdm

from models.decompose_field import DensityVM, AppVM
from models.tensorBase import *
from models.renderer import *
from models.myutils import *
from models.tensoIR.relight_utils import *
from utils import TVLoss, visualize_depth_numpy
from models.volrend import rendering


class TensoIR(TensorBase):
    def setup(self):
        self.light_kind = self.config.light.light_kind
        self.envmap_w = self.config.light.envmap_w
        self.envmap_h = self.config.light.envmap_h
        self.numLgtSGs = self.config.light.numLgtSGs
        # self.light_num = len(self.config.light.rotation)
        self.light_num = 1
        self.light_rotation = self.config.light.rotation
        self.normals_kind = self.config.normals_kind
        self.fixed_fresnel = self.config.fixed_fresnel
        self.tvreg = TVLoss()
        self.is_relight = False

        super(TensoIR, self).setup()

        self.init_light()

    def init_render_func(self, app_dim, conf):
        super(TensoIR, self).init_render_func(app_dim, conf)

        if self.normals_kind in ["purely_predicted", "derived_plus_predicted"]:
            self.renderModule_normal = MLPBRDF_PEandFeature(app_dim,
                conf.pos_pe, conf.fea_pe, conf.featureC, outc=3, act_net=nn.Tanh())
        elif self.normals_kind == "residue_prediction":
            self.renderModule_normal = MLPNormal_normal_and_PExyz(app_dim,
                conf.pos_pe, conf.fea_pe, conf.featureC, outc=3, act_net=nn.Tanh())

        # 4 = 3 + 1: albedo + roughness
        self.renderModule_brdf= MLPBRDF_PEandFeature(app_dim,
            conf.pos_pe, conf.fea_pe, conf.featureC, outc=4, act_net=nn.Sigmoid())
        print("renderModule_brdf", self.renderModule_brdf)

    def generate_envir_map_dir(self, envmap_h, envmap_w, is_jittor=False):
        lat_step_size = np.pi / envmap_h
        lng_step_size = 2 * np.pi / envmap_w
        phi, theta = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, envmap_h),
                                    torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, envmap_w)], indexing='ij')

        sin_phi = torch.sin(torch.pi / 2 - phi)  # [envH, envW]
        light_area_weight = 4 * torch.pi * sin_phi / torch.sum(sin_phi)  # [envH, envW]
        assert 0 not in light_area_weight, "There shouldn't be light pixel that doesn't contribute"
        light_area_weight = light_area_weight.to(torch.float32).reshape(-1) # [envH * envW, ]
        if is_jittor:
            phi_jittor, theta_jittor = lat_step_size * (torch.rand_like(phi) - 0.5),  lng_step_size * (torch.rand_like(theta) - 0.5)
            phi, theta = phi + phi_jittor, theta + theta_jittor

        view_dirs = torch.stack([torch.cos(theta) * torch.cos(phi),
                                 torch.sin(theta) * torch.cos(phi),
                                 torch.sin(phi)], dim=-1).view(-1, 3)    # [envH * envW, 3]

        return light_area_weight, view_dirs

    def init_light(self):
        self.light_area_weight, self.fixed_viewdirs = self.generate_envir_map_dir(self.envmap_h, self.envmap_w)
        nlights = self.envmap_w * self.envmap_h

        if self.light_kind == 'pixel':
            self._light_rgbs = torch.nn.Parameter(torch.FloatTensor(nlights, 3).uniform_(0, 3).to(torch.float32)) # [envH * envW, 3]
        elif self.light_kind == 'sg':
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True)  # [M, 7]; lobe + lambda + mu
            self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))
            # make sure lambda is not too close to zero
            self.lgtSGs.data[:, 3:4] = 10. + torch.abs(self.lgtSGs.data[:, 3:4] * 20.)
            # init envmap energy
            energy = compute_energy(self.lgtSGs.data)
            self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi * 0.8
            energy = compute_energy(self.lgtSGs.data)
            print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

            # deterministicly initialize lobes
            lobes = fibonacci_sphere(self.numLgtSGs//2).astype(np.float32)
            self.lgtSGs.data[:self.numLgtSGs//2, :3] = torch.from_numpy(lobes)
            self.lgtSGs.data[self.numLgtSGs//2:, :3] = torch.from_numpy(lobes)

        # rotation matrixs for incident light
        self.light_rotation_matrix = []
        for i in range(self.light_num):
            horizontal_angle = torch.tensor(self.light_rotation[i] / 180 * torch.pi).to(torch.float32)
            rotation_matrix = torch.tensor(
                [[torch.cos(horizontal_angle), -torch.sin(horizontal_angle), 0],
                 [torch.sin(horizontal_angle), torch.cos(horizontal_angle), 0],
                 [0, 0, 1]]).to(torch.float32)

            self.light_rotation_matrix.append(rotation_matrix)
        self.light_rotation_matrix = torch.stack(self.light_rotation_matrix, dim=0) # [rotation_num, 3, 3]

    def init_svd_volume(self, args):
        self.density = DensityVM(args.density.n_comp, self.grid_size)
        self.app = AppVM(args.app.n_comp, self.grid_size, args.app.feature_dim)
        self.light_line = nn.Embedding(self.light_num, sum(self.app.n_comp))
        # (light_num, sum(self.app_n_comp)), such as (10, 16+16+16)

    def get_optparam_groups(self, conf, lr_scale=1.0):
        grad_vars = [{'params': self.density.line, 'lr': conf.lr_xyz * lr_scale},
                     {'params': self.density.plane, 'lr': conf.lr_xyz * lr_scale},
                     {'params': self.app.line, 'lr': conf.lr_xyz * lr_scale},
                     {'params': self.app.plane, 'lr': conf.lr_xyz * lr_scale},
                     {'params': self.app.mat.parameters(), 'lr': conf.lr_net * lr_scale},
                    #  {'params': self.light_line.parameters(), 'lr': 0.001},
                    ]

        if self.light_kind == 'pixel':
            grad_vars += [{'params': self._light_rgbs, 'lr': 0.001}]
        elif self.light_kind == 'sg':
            grad_vars += [{'params': self.lgtSGs, 'lr': 0.001}]

        if isinstance(self.renderModule, nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': conf.lr_net}]

        if isinstance(self.renderModule_brdf, nn.Module):
            grad_vars += [{'params': self.renderModule_brdf.parameters(), 'lr':conf.lr_net}]

        if (self.normals_kind in
            ["purely_predicted", "derived_plus_predicted" "residue_prediction"]) \
                and isinstance(self.renderModule_normal, nn.Module):
            grad_vars += [{'params': self.renderModule_normal.parameters(), 'lr':conf.lr_net}]

        return grad_vars

    def vector_comp_diffs(self):
        return self.density.vectorDiff() + self.app.vectorDiff()

    def compute_bothfeature(self, xyz_sampled, light_idx=None):
        app_feature = self.app.compute(xyz_sampled)
        return app_feature, app_feature

    def compute_intrinfeature(self, xyz_sampled):
        return self.app.compute(xyz_sampled)

    def compute_intrinfeature_with_grad(self, xyz_sampled):
        return self.app.compute_with_grad(xyz_sampled)

    def compute_appfeature(self, xyz_sampled, light_idx=None):
        return self.app.compute(xyz_sampled)

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        render_step_size = self.render_step_size
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = render_step_size * rng.to(rays_o.device)
        interpx = (t_min[..., None] + step)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    @torch.enable_grad()
    def compute_derived_normals(self, xyz_locs):
        xyz_locs.requires_grad_(True)
        sigma_feature = self.compute_densityfeature_with_grad(xyz_locs)  # [..., 1]  detach() removed in the this function
        sigma = self.feature2density(sigma_feature)
        d_output = torch.ones_like(sigma, requires_grad=False, device=sigma.device)

        gradients = torch.autograd.grad(
                                    outputs=sigma,
                                    inputs=xyz_locs,
                                    grad_outputs=d_output,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True
                                    )[0]
        derived_normals = -F.normalize(gradients, p=2, dim=-1, eps=1e-6)
        derived_normals = derived_normals.view(-1, 3)
        return derived_normals

    def compute_relative_smoothness_loss(self, values, values_jittor):
        base = torch.maximum(values, values_jittor).clip(min=1e-6)
        difference = torch.sum(((values - values_jittor) / base)**2, dim=-1, keepdim=True)  # [..., 1]
        return difference

    def get_light_rgbs(self, incident_light_directions, device='cuda'):
        '''
        - args:
            - incident_light_directions: [sample_number, 3]
        - return:
            - light_rgbs: [rotation_num, sample_number, 3]
        '''
        init_light_directions = incident_light_directions.to(device).reshape(1, -1, 3) # [1, sample_number, 3]
        rotation_matrix = self.light_rotation_matrix.to(device) # [rotation_num, 3, 3]
        remapped_light_directions = torch.matmul(init_light_directions, rotation_matrix).reshape(-1, 3) # [rotation_num * sample_number, 3]
        if self.light_kind == 'sg':
            light_rgbs = render_envmap_sg(self.lgtSGs.to(device), remapped_light_directions).reshape(self.light_num, -1, 3) # [rotation_num, sample_number, 3]
        else:
            if self.light_kind == 'pixel':
                environment_map = torch.nn.functional.softplus(self._light_rgbs, beta=5).reshape(self.envmap_h, self.envmap_w, 3).to(device) # [H, W, 3]
            # elif self.light_kind == 'gt':
            #     environment_map = self.dataset.lights_probes.requires_grad_(False).reshape(self.envmap_h, self.envmap_w, 3).to(device) # [H, W, 3]
            else:
                print("Illegal light kind: {}".format(self.light_kind))
                exit(1)
            environment_map = environment_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
            phi = torch.arccos(remapped_light_directions[:, 2]).reshape(-1) - 1e-6
            theta = torch.atan2(remapped_light_directions[:, 1], remapped_light_directions[:, 0]).reshape(-1)
            # normalize to [-1, 1]
            query_y = (phi / np.pi) * 2 - 1
            query_x = - theta / np.pi
            grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)

            light_rgbs = F.grid_sample(environment_map, grid, align_corners=False).squeeze().permute(1, 0).reshape(self.light_num, -1, 3)
        return light_rgbs

    def gen_light_incident_dirs(self, sample_number=-1, method='fixed_envirmap', device='cuda'):

        ''' This function is used to generate light incident directions per iteraration,
            and this function is used for the light kind of 'sg'
        - args:
            - sample_number: sampled incident light directions, this argumet is not always used
            - method:
                    'fixed_envirmap': generate light incident directions on the fixed center points of the environment map
                    'uniform_sample': sample incident direction uniformly on the unit sphere, sample number is specified by sample_number
                    'stratified_sampling': random sample incident direction on each grid of envirment map
                    'importance_sample': sample based on light energy
        - return:
            - light_incident_directions: [out_putsample_number, 3]
        '''
        if method == 'fixed_envirmap':
            light_incident_directions = self.fixed_viewdirs
        elif method == 'uniform_sample':
            # uniform sampling 'sample_number' points on a unit sphere
            pass # TODO
        elif method == 'stratified_sampling':
            lat_step_size = np.pi / self.envmap_h
            lng_step_size = 2 * np.pi / self.envmap_w

            phi_begin, theta_begin = torch.meshgrid([
                                        torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, self.envmap_h),
                                        torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, self.envmap_w)
                                        ],
                                        indexing='ij')
            phi_jittor, theta_jittor = lat_step_size * (torch.rand_like(phi_begin) - 0.5),  lng_step_size * (torch.rand_like(theta_begin) - 0.5)

            phi, theta = phi_begin + phi_jittor, theta_begin + theta_jittor

            light_incident_directions = torch.stack([torch.cos(theta) * torch.cos(phi),
                                        torch.sin(theta) * torch.cos(phi),
                                        torch.sin(phi)], dim=-1)    # [H, W, 3]

        elif method == 'stratifed_sample_equal_areas':

            sin_phi_size = 2 / self.envmap_h
            lng_step_size = 2 * np.pi / self.envmap_w


            sin_phi_begin, theta_begin = torch.meshgrid([torch.linspace(1 - 0.5 * sin_phi_size, -1 + 0.5 * sin_phi_size, self.envmap_h),
                                                        torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, self.envmap_w)], indexing='ij')

            sin_phi_jittor, theta_jittor = sin_phi_size * (torch.rand_like(sin_phi_begin) - 0.5),  lng_step_size * (torch.rand_like(theta_begin) - 0.5)

            sin_phi, theta = sin_phi_begin + sin_phi_jittor, theta_begin + theta_jittor

            phi = torch.asin(sin_phi)
            light_incident_directions = torch.stack([torch.cos(theta) * torch.cos(phi),
                            torch.sin(theta) * torch.cos(phi),
                            torch.sin(phi)], dim=-1)    # [H, W, 3]


        elif method == 'importance_sample':
            _, view_dirs = self.generate_envir_map_dir(128, 256, is_jittor=True)
            envir_map = self.get_light_rgbs(view_dirs.reshape(-1, 3).to(device))[0]
            with torch.no_grad():
                envir_map = envir_map.reshape(128, 256, 3)

                # compute the pdf of importance sampling of the environment map
                light_intensity = torch.sum(envir_map, dim=2, keepdim=True) # [H, W, 1]
                env_map_h, env_map_w, _ = light_intensity.shape
                h_interval = 1.0 / env_map_h
                sin_theta = torch.sin(torch.linspace(0 + 0.5 * h_interval, np.pi - 0.5 * h_interval, env_map_h)).to(device) # [H, ]
                pdf = light_intensity * sin_theta.view(-1, 1, 1) # [H, W, 1]
                pdf_to_sample = pdf / torch.sum(pdf)  # [H, W, 1]
                pdf_to_compute = pdf_to_sample * env_map_h * env_map_w / (2 * np.pi * np.pi * sin_theta.view(-1, 1, 1))

                light_dir_idx = torch.multinomial(pdf_to_sample.view(-1), sample_number, replacement=True) # [sample_number, ]
                envir_map_dir = view_dirs.view(-1, 3).to(device)

                light_dir = envir_map_dir.gather(0, light_dir_idx.unsqueeze(-1).expand(-1, 3)).view(-1, 3) # [num_samples, 3]
                # sample the light rgbs
                envir_map_rgb = envir_map.view(-1, 3)
                light_rgb = envir_map_rgb.gather(0, light_dir_idx.unsqueeze(-1).expand(-1, 3)).view(-1, 3) # [num_samples, 3]
                envir_map_pdf = pdf_to_compute.view(-1, 1)
                light_pdf = envir_map_pdf.gather(0, light_dir_idx.unsqueeze(-1).expand(-1, 1)).view(-1, 1) # [num_samples, 1]

                return light_dir, light_rgb, light_pdf

        return light_incident_directions.reshape(-1, 3) # [output_sample_number, 3]

    def cal_rgb(self, positions, ray_indices, dists):
        self.ray_indices = ray_indices
        t_dirs = self.rays_d[ray_indices]
        # t_idx = self.light_idx[ray_indices]
        positions = self.normalize_coord(positions)
        sigmas = self.compute_density(positions)
        radiance_field_feat, intrinsic_feat = self.compute_bothfeature(positions)
        rgbs = self.renderModule(positions, t_dirs, radiance_field_feat)

        if self.is_relight:
            brdf = self.renderModule_brdf(positions, intrinsic_feat)
            self.albedo, self.roughness = brdf[..., :3], (brdf[..., 3:4] * 0.9 + 0.09)

            positions_jittor = positions + torch.randn_like(positions) * 0.01
            intrinsic_feat_jittor = self.compute_intrinfeature(positions_jittor)
            brdf_jittor = self.renderModule_brdf(positions_jittor, intrinsic_feat_jittor)
            albedo_jittor, roughness_jittor = brdf_jittor[..., :3], (brdf_jittor[..., 3:4] * 0.9 + 0.09)

            self.albedo_smoothness_cost = self.compute_relative_smoothness_loss(self.albedo, albedo_jittor)
            self.roughness_smoothness_cost = self.compute_relative_smoothness_loss(self.roughness, roughness_jittor)

            derived_normals = self.compute_derived_normals(positions)
            predicted_normals = self.renderModule_normal(positions, intrinsic_feat)
            self.normals_diff = torch.sum(torch.pow(derived_normals - predicted_normals, 2), dim=-1, keepdim=True)
            self.normals_orientation_loss = torch.sum(t_dirs * predicted_normals, dim=-1, keepdim=True).clamp(min=0)
            if self.normals_kind == 'purely_predicted':
                valid_normals = predicted_normals
            elif self.normals_kind == 'purely_derived':
                valid_normals = derived_normals
            elif self.normals_kind == 'gt_normals':
                valid_normals = torch.zeros_like(predicted_normals)
            elif self.normals_kind == 'derived_plus_predicted':
                valid_normals = predicted_normals
            else:
                raise NotImplementedError(f"normals_kind {self.normals_kind} not implemented")
            self.normal = valid_normals

        return rgbs, sigmas

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
            rgb_sigma_fn=self.rgb_fn,
            render_bkgd=self.background_color,
        )

        return rgb, opacity, depth, extras

    def myforward(self, rays, args, n_rays):
        rgb_map, acc_map, depth_map, extras = self.forward(rays, is_train=True)

        weights = extras['weights']
        app_mask = weights > self.alphaMask_thres
        # print('app_mask', app_mask.shape, app_mask.sum())

        if not self.is_relight:
            rgb_with_brdf = torch.ones_like(rgb_map)
            normal_map = None
            albedo_map = None
            roughness_map = None
            fresnel_map = None
            normals_diff_map = None
            normals_orientation_loss_map = None
            albedo_smoothness_loss = None
            roughness_smoothness_loss = None
        else:
            # print("output now gpu memory")
            # print(torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")
            weights = extras['weights']
            if weights.shape[0] == 0:
                self.ray_indices = torch.zeros((0,)).long().to(self.device)
                self.albedo = torch.zeros((0, 3)).to(self.device)
                self.roughness = torch.zeros((0, 1)).to(self.device)
                self.albedo_smoothness_cost = torch.zeros((0, 1)).to(self.device)
                self.roughness_smoothness_cost = torch.zeros((0, 1)).to(self.device)
                self.normals_diff = torch.zeros((0, 1)).to(self.device)
                self.normals_orientation_loss = torch.zeros((0, 1)).to(self.device)
                self.normal = torch.zeros((0, 3)).to(self.device)

            normal_map = accumulate_along_rays(
                weights, self.normal, self.ray_indices, n_rays)
            normals_diff_map = accumulate_along_rays(
                weights, self.normals_diff, self.ray_indices, n_rays)
            normals_orientation_loss_map = accumulate_along_rays(
                weights, self.normals_orientation_loss, self.ray_indices, n_rays)
            albedo_map = accumulate_along_rays(
                weights, self.albedo, self.ray_indices, n_rays)
            roughness_map = accumulate_along_rays(
                weights, self.roughness, self.ray_indices, n_rays)
            fresnel_map = torch.zeros_like(albedo_map).fill_(self.fixed_fresnel)
            albedo_smoothness_cost_map = accumulate_along_rays(
                weights, self.albedo_smoothness_cost, self.ray_indices, n_rays)
            roughness_smoothness_cost_map = accumulate_along_rays(
                weights, self.roughness_smoothness_cost, self.ray_indices, n_rays)

            albedo_smoothness_loss = torch.mean(albedo_smoothness_cost_map)
            roughness_smoothness_loss = torch.mean(roughness_smoothness_cost_map)

            if args.white_bg or torch.rand((1,)) < 0.5:
                normal_map = normal_map + (1 - acc_map) * torch.tensor(
                    [0.0, 0.0, 1.0], device=normal_map.device)  # Background normal
                # normal_map = normal_map

                albedo_map = albedo_map + (1 - acc_map)  # Albedo background should be white
                roughness_map = roughness_map + (1 - acc_map)
                fresnel_map = fresnel_map + (1 - acc_map)

            # tone mapping & gamma correction
            rgb_map = rgb_map.clamp(0, 1)
            # Tone mapping to make sure the output of self.renderModule() is in linear space,
            # and the rgb_map output of this forward() is in sRGB space.
            # By doing this, we can use the output of self.renderModule() to better
            # represent the indirect illumination, which is implemented in another function.
            if rgb_map.shape[0] > 0:
                rgb_map = linear2srgb_torch(rgb_map)

            albedo_map = albedo_map.clamp(0, 1)
            fresnel_map = fresnel_map.clamp(0, 1)
            roughness_map = roughness_map.clamp(0, 1)
            normal_map = F.normalize(normal_map, p=2, dim=-1, eps=1e-6)

            acc_mask = acc_map > 0.5 # where there may be intersected surface points

            acc_mask = acc_mask.squeeze(-1)
            depth_map = depth_map.squeeze(-1)
            pos = torch.zeros_like(normal_map)
            rgb_with_brdf_masked, pos[acc_mask] = render_with_BRDF(
                depth_map[acc_mask],
                normal_map[acc_mask],
                albedo_map[acc_mask],
                roughness_map[acc_mask].repeat(1, 3),
                fresnel_map[acc_mask],
                rays[acc_mask],
                self,
                # self.light_idx[acc_mask],
                sample_method=args.light_sample_train,
                chunk_size=args.relight_chunk_size,
                device=self.device,
                args=args,
            )
            rgb_with_brdf = torch.ones_like(rgb_map) # background default to be white
            rgb_with_brdf[acc_mask] = rgb_with_brdf_masked
            # rgb_with_brdf = rgb_with_brdf * acc_map  + (1. - acc_map)

        return {
            "rgb_map": rgb_map,
            "depth_map": depth_map,
            "normal_map": normal_map,
            "albedo_map": albedo_map,
            "acc_map": acc_map,
            "roughness_map": roughness_map,
            "fresnel_map": fresnel_map,
            'rgb_with_brdf_map': rgb_with_brdf,
            'normals_diff_map': normals_diff_map,
            'normals_orientation_loss_map': normals_orientation_loss_map,
            'albedo_smoothness_loss': albedo_smoothness_loss,
            'roughness_smoothness_loss': roughness_smoothness_loss,
        }

    def cal_loss(self, data, args):
        rays = data['rays']
        n_rays = rays.shape[0]
        rgb_gt = data['rgbs']
        light_idx = data['light_idx']
        self.light_idx = light_idx
        device = rays.device
        self.is_relight = args.relight_flag

        ret = self.myforward(rays, args, n_rays)

        loss_dict = {}

        total_loss = 0
        loss_rgb_brdf = torch.tensor(1e-6).to(device)
        loss_rgb = torch.mean((ret['rgb_map'] - rgb_gt) ** 2)
        total_loss += loss_rgb
        loss_dict['loss_rgb'] = loss_rgb

        if self.ortho_reg_weight > 0:
            loss_reg = self.vector_comp_diffs()
            total_loss += self.ortho_reg_weight * loss_reg
            loss_dict['loss_reg'] = loss_reg

        if self.l1_reg_weight > 0:
            loss_reg_L1 = self.density_L1()
            total_loss += self.l1_reg_weight * loss_reg_L1
            loss_dict['loss_reg_L1'] = loss_reg_L1

        if self.tv_weight_density > 0:
            loss_tv = self.TV_loss_density(self.tvreg) * self.tv_weight_density
            total_loss = total_loss + loss_tv
            loss_dict['loss_tv_density'] = loss_tv

        if self.tv_weight_app > 0:
            loss_tv = self.TV_loss_app(self.tvreg) * self.tv_weight_app
            total_loss = total_loss + loss_tv
            loss_dict['loss_tv_app'] = loss_tv

        if args.relight_flag:
            loss_rgb_brdf = torch.mean((ret['rgb_with_brdf_map'] - rgb_gt) ** 2)
            loss_dict['loss_rgb_brdf'] = loss_rgb_brdf
            loss_dict['PSNR_brdf'] = -10.0 * torch.log10(loss_rgb_brdf)
            total_loss += loss_rgb_brdf * args.rgb_brdf_weight

            # exponential growth
            normal_weight_factor = args.normals_loss_enhance_ratio ** ((args.nw_iter - args.update_AlphaMask_list[0])/ (args.iteration - args.update_AlphaMask_list[0]))
            BRDF_weight_factor = args.BRDF_loss_enhance_ratio ** ((args.nw_iter - args.update_AlphaMask_list[0])/ (args.iteration - args.update_AlphaMask_list[0]))

            if args.normals_diff_weight > 0:
                loss_normals_diff = normal_weight_factor * args.normals_diff_weight * ret['normals_diff_map'].mean()
                total_loss += loss_normals_diff
                loss_dict['loss_normals_diff'] = loss_normals_diff

            if args.normals_orientation_weight > 0:
                loss_normals_orientation = normal_weight_factor * args.normals_orientation_weight * ret['normals_orientation_loss_map'].mean()
                total_loss += loss_normals_orientation
                loss_dict['loss_normals_orientation'] = loss_normals_orientation

            if args.roughness_smoothness_loss_weight > 0:
                roughness_smoothness_loss = BRDF_weight_factor * args.roughness_smoothness_loss_weight * ret['roughness_smoothness_loss']
                total_loss += roughness_smoothness_loss
                loss_dict['loss_roughness_smoothness'] = roughness_smoothness_loss

            if args.albedo_smoothness_loss_weight > 0:
                albedo_smoothness_loss = BRDF_weight_factor * args.albedo_smoothness_loss_weight * ret['albedo_smoothness_loss']
                total_loss += albedo_smoothness_loss
                loss_dict['loss_albedo_smoothness'] = albedo_smoothness_loss

        loss_dict['total_loss'] = total_loss

        loss_dict['PSNR'] = -10.0 * torch.log10(loss_rgb)
        return loss_dict

    def evaluation(
        self,
        dataset,
        args,
        device,
        savePath=None,
        N_vis=-1,
        prefix='',
        lst='',
    ):

        PSNRs_rgb, rgb_maps, depth_maps, gt_maps, gt_rgb_brdf_maps = [], [], [], [], []
        PSNRs_rgb_brdf = []
        rgb_with_brdf_maps, normal_rgb_maps, normal_rgb_vis_maps = [], [], []
        albedo_maps, albedo_gamma_maps, roughness_maps, fresnel_maps, normals_diff_maps, normals_orientation_loss_maps = [], [], [], [], [], []
        ssims, l_alex, l_vgg = [], [], []
        ssims_rgb_brdf, l_alex_rgb_brdf, l_vgg_rgb_brdf = [], [], []

        if savePath is not None:
            os.makedirs(savePath, exist_ok=True)
            os.makedirs(savePath + "/nvs_with_radiance_field", exist_ok=True)
            os.makedirs(savePath + "/nvs_with_brdf", exist_ok=True)
            os.makedirs(savePath + "/normal", exist_ok=True)
            os.makedirs(savePath + "/normal_vis", exist_ok=True)
            os.makedirs(savePath + "/brdf", exist_ok=True)
            os.makedirs(savePath + "/envir_map/", exist_ok=True)
            os.makedirs(savePath + "/acc_map", exist_ok=True)

        near_far = dataset.near_far
        W, H = dataset.img_wh

        _, view_dirs = self.generate_envir_map_dir(256, 512)

        predicted_envir_map = self.get_light_rgbs(view_dirs.reshape(-1, 3).to(device))[0]
        predicted_envir_map = predicted_envir_map.reshape(256, 512, 3).cpu().detach().numpy()
        predicted_envir_map = np.clip(predicted_envir_map, a_min=0, a_max=np.inf)
        predicted_envir_map = np.uint8(np.clip(np.power(predicted_envir_map, 1./2.2), 0., 1.) * 255.)
        envirmap = predicted_envir_map

        # save predicted envir map
        imageio.imwrite(f'{savePath}/envir_map/{prefix}envirmap.png', envirmap)

        img_eval_interval = 1 if N_vis < 0 else max(dataset.all_rays.shape[0] // N_vis, 1)
        idxs = list(range(0, dataset.all_rays.shape[0], img_eval_interval))

        with torch.no_grad():
            self.eval()
            for idx in tqdm(idxs):
                data = dataset.__getitem__(idx)
                data['rays'] = data['rays'].view(-1, data['rays'].shape[-1])

                rgb_map, acc_map, depth_map, normal_map, albedo_map, roughness_map, albedo_gamma_map = [], [], [], [], [], [], []
                fresnel_map, rgb_with_brdf_map, normals_diff_map, normals_orientation_loss_map = [], [], [], []
                chunk = 4096
                N_rays = data['rays'].shape[0]

                pos_map = []
                dir_map = []

                for i in tqdm(range(N_rays // chunk + int(N_rays % chunk > 0))):
                    nw_rays = data['rays'][i * chunk:(i + 1) * chunk].to(device)
                    self.light_idx = data['light_idx'][i * chunk:(i + 1) * chunk].to(device)
                    ret = self.myforward(nw_rays, args, n_rays=nw_rays.shape[0])
                    rgb_map.append(ret['rgb_map'].detach().cpu())
                    depth_map.append(ret['depth_map'].detach().cpu())
                    normal_map.append(ret['normal_map'].detach().cpu())
                    albedo_map.append(ret['albedo_map'].detach().cpu())
                    acc_map.append(ret['acc_map'].detach().cpu())
                    roughness_map.append(ret['roughness_map'].detach().cpu())
                    fresnel_map.append(ret['fresnel_map'].detach().cpu())
                    rgb_with_brdf_map.append(ret['rgb_with_brdf_map'].detach().cpu())
                    normals_diff_map.append(ret['normals_diff_map'].detach().cpu())
                    normals_orientation_loss_map.append(ret['normals_orientation_loss_map'].detach().cpu())

                rgb_map = torch.cat(rgb_map)
                depth_map = torch.cat(depth_map)
                normal_map = torch.cat(normal_map)
                albedo_map = torch.cat(albedo_map)
                roughness_map = torch.cat(roughness_map)
                fresnel_map = torch.cat(fresnel_map)
                rgb_with_brdf_map = torch.cat(rgb_with_brdf_map)
                normals_diff_map = torch.cat(normals_diff_map)
                normals_orientation_loss_map = torch.cat(normals_orientation_loss_map)
                acc_map = torch.cat(acc_map)

                rgb_map = rgb_map.clamp(0.0, 1.0)
                rgb_with_brdf_map = rgb_with_brdf_map.clamp(0.0, 1.0)

                acc_map = acc_map.reshape(H, W).detach().cpu()

                rgb_map, depth_map = rgb_map.reshape(H, W, 3).detach().cpu(), depth_map.reshape(H, W).detach().cpu()
                rgb_with_brdf_map = rgb_with_brdf_map.reshape(H, W, 3).detach().cpu()
                albedo_map = albedo_map.reshape(H, W, 3).detach().cpu()

                albedo_gamma_map = (albedo_map.clip(0, 1.)) ** (1.0 / 2.2)

                roughness_map = roughness_map.reshape(H, W, 1).repeat(1, 1, 3).detach().cpu()
                fresnel_map = fresnel_map.reshape(H, W, 3).detach().cpu()
                depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

                gt_rgb = data['rgbs'].view(H, W, 3)
                gt_rgb_wirh_brdf = gt_rgb.clone()
                loss_rgb = torch.mean((rgb_map - gt_rgb) ** 2)
                loss_rgb_brdf = torch.mean((rgb_with_brdf_map - gt_rgb_wirh_brdf) ** 2)
                print('rgb_with_brdf_map', rgb_with_brdf_map.dtype)
                print('loss_rgb_brdf', loss_rgb_brdf.dtype, loss_rgb_brdf.item())
                PSNRs_rgb.append(-10.0 * np.log(loss_rgb.item()) / np.log(10.0))
                PSNRs_rgb_brdf.append(10 * torch.log10(1 / loss_rgb_brdf))
                print('PSNRs_rgb_brdf', PSNRs_rgb_brdf[-1])

                rgb_map = (rgb_map.numpy() * 255).astype('uint8')
                rgb_with_brdf_map = (rgb_with_brdf_map.numpy() * 255).astype('uint8')
                gt_rgb = (gt_rgb.numpy() * 255).astype('uint8')
                gt_rgb_wirh_brdf = (gt_rgb_wirh_brdf.numpy() * 255).astype('uint8')
                albedo_map = (albedo_map.numpy() * 255).astype('uint8')
                albedo_gamma_map = (albedo_gamma_map.numpy() * 255).astype('uint8')
                roughness_map = (roughness_map.numpy() * 255).astype('uint8')
                fresnel_map = (fresnel_map.numpy() * 255).astype('uint8')
                acc_map = (acc_map.numpy() * 255).astype('uint8')

                # Visualize normal
                ## Prediction
                normal_map = F.normalize(normal_map, dim=-1)

                normal_rgb_map = normal_map * 0.5 + 0.5 # map from [-1, 1] to [0, 1] to visualize
                normal_rgb_map = (normal_rgb_map.reshape(H, W, 3).cpu().numpy() * 255).astype('uint8')
                normal_rgb_vis_map = (normal_rgb_map * (acc_map[:, :, None] / 255.0) + (1 -(acc_map[:, :, None] / 255.0)) * 255).astype('uint8') # white background


                # difference between the predicted normals and derived normals
                normals_diff_map = (torch.clamp(normals_diff_map, 0.0, 1.0).reshape(H, W, 1).repeat(1, 1, 3).numpy() * 255).astype('uint8')

                # normals orientation loss map
                normals_orientation_loss_map = (torch.clamp(normals_orientation_loss_map , 0.0, 1.0).reshape(H, W, 1).repeat(1, 1, 3).numpy() * 255).astype('uint8')

                rgb_maps.append(rgb_map)
                rgb_with_brdf_maps.append(rgb_with_brdf_map)
                depth_maps.append(depth_map)
                gt_maps.append(gt_rgb)
                gt_rgb_brdf_maps.append(gt_rgb_wirh_brdf)
                normal_rgb_maps.append(normal_rgb_map)
                normal_rgb_vis_maps.append(normal_rgb_vis_map)

                normals_diff_maps.append(normals_diff_map)
                normals_orientation_loss_maps.append(normals_orientation_loss_map)

                albedo_maps.append(albedo_map)
                albedo_gamma_maps.append(albedo_gamma_map)
                roughness_maps.append(roughness_map)
                fresnel_maps.append(fresnel_map)


                if savePath is not None:
                    rgb_map = np.concatenate((rgb_map, gt_rgb, depth_map), axis=1)
                    rgb_with_brdf_map = np.concatenate((rgb_with_brdf_map, gt_rgb_wirh_brdf), axis=1)

                    normal_map = np.concatenate((normal_rgb_map, normals_diff_map, normals_orientation_loss_map), axis=1)
                    brdf_map = np.concatenate((albedo_map, roughness_map, fresnel_map), axis=1)

                    imageio.imwrite(f'{savePath}/nvs_with_radiance_field/{prefix}{idx:03d}.png', rgb_map)
                    imageio.imwrite(f'{savePath}/nvs_with_brdf/{prefix}{idx:03d}.png', rgb_with_brdf_map)
                    imageio.imwrite(f'{savePath}/normal/{prefix}{idx:03d}.png', normal_map)
                    imageio.imwrite(f'{savePath}/normal_vis/{prefix}{idx:03d}.png', normal_rgb_vis_map)
                    imageio.imwrite(f'{savePath}/brdf/{prefix}{idx:03d}.png', brdf_map)
                    imageio.imwrite(f'{savePath}/brdf/{prefix}{idx:03d}_albedo.png', albedo_gamma_map)
                    imageio.imwrite(f'{savePath}/brdf/{prefix}{idx:03d}_roughness.png', roughness_map)
                    imageio.imwrite(f'{savePath}/acc_map/{prefix}{idx:03d}.png', acc_map)

            self.train()

        # Compute metrics
        if PSNRs_rgb:
            psnr = np.mean(np.asarray(PSNRs_rgb))
            psnr_rgb_brdf = np.mean(np.asarray(PSNRs_rgb_brdf))

            saved_message = f'Iteration:{prefix[:-1]}, PSNR_nvs: {psnr:.2f}, PSNR_nvs_brdf: {psnr_rgb_brdf:.2f}\n'
            # write the end of record file
            with open(f'{savePath}/metrics_record.txt', 'a') as f:
                f.write(saved_message)

        return PSNRs_rgb