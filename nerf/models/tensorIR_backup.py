from models.tensorbase_backup import *
from models.renderer import *
from models.myutils import *
from models.tensoIR.relight_utils import *
from utils import TVLoss


class TensoIR(TensorBase):
    def __init__(self, args, device, aabb, gridSize):
        self.light_kind = args.light.light_kind
        self.envmap_w = args.light.envmap_w
        self.envmap_h = args.light.envmap_h
        self.numLgtSGs = args.light.numLgtSGs
        self.light_num = len(args.light.rotation)
        self.light_rotation = args.light.rotation

        self.normals_kind = args.normals_kind
        self.fixed_fresnel = args.fixed_fresnel

        super(TensoIR, self).__init__(args, device, aabb, gridSize)

        self.tvreg = TVLoss()
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
        self.density = DensityVM(args.density.n_comp, self.gridSize)
        self.app = AppVM(args.app.n_comp, self.gridSize, args.app.feature_dim)
        self.light_line = torch.nn.Embedding(self.light_num, sum(self.app.n_comp))
        # (light_num, sum(self.app_n_comp)), such as (10, 16+16+16)

    def get_optparam_groups(self, conf, lr_scale=1.0):
        grad_vars = [{'params': self.density.line, 'lr': conf.lr_xyz * lr_scale},
                     {'params': self.density.plane, 'lr': conf.lr_xyz * lr_scale},
                     {'params': self.app.line, 'lr': conf.lr_xyz * lr_scale},
                     {'params': self.app.plane, 'lr': conf.lr_xyz * lr_scale},
                     {'params': self.app.mat.parameters(), 'lr': conf.lr_net * lr_scale},
                     {'params': self.light_line.parameters(), 'lr': 0.001}]

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

    def compute_bothfeature(self, xyz_sampled, light_idx):
        app_feature = self.app.compute_feature(xyz_sampled) # (sampled_pts, sum(self.app_n_comp))

        light_coef_point = self.light_line(light_idx.to(self.device)).squeeze(1).permute(1,0)
        radiance_field_feat = self.app.mat((app_feature * light_coef_point).T)

        static_index = torch.arange(self.light_num).to(self.device, dtype=torch.int32)  # [light_num, ]
        mean_weight = torch.mean(self.light_line(static_index), dim=0).unsqueeze(-1).expand_as(light_coef_point)
        intrinsic_feat = self.app.mat((app_feature * mean_weight).T)

        return radiance_field_feat, intrinsic_feat

    def compute_intrinfeature(self, xyz_sampled):
        app_feature = self.app.compute_feature(xyz_sampled) # (sampled_pts, sum(self.app_n_comp))
        static_index = torch.arange(self.light_num).to(xyz_sampled.device, dtype=torch.int32) # [light_num, ]
        mean_weight = torch.mean(self.light_line(static_index), dim=0).unsqueeze(-1).expand_as(app_feature)
        intrinsic_feat = self.app.mat((app_feature * mean_weight).T)
        return intrinsic_feat

    def compute_appfeature(self, xyz_sampled, light_idx):
        app_feature = self.app.compute_feature(xyz_sampled) # (sampled_pts, sum(self.app_n_comp))
        light_coef_point = self.light_line(light_idx.to(self.device)).squeeze(1).permute(1,0)
        radiance_field_feat = self.app.mat((app_feature * light_coef_point).T)
        return radiance_field_feat

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
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[..., None] + step)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def get_mid_and_interval(self, batch_size, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples

        s = torch.linspace(0, 1, N_samples+1).cuda()
        m = (s[1:] + s[:-1]) * 0.5
        m = m[None].repeat(batch_size,1)
        interval = 1 / N_samples
        return m , interval

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

    def get_light_rgbs(self, incident_light_directions=None, device='cuda'):
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

    def forward(
        self,
        rays_chunk,
        light_idx,
        white_bg=True,
        is_train=False,
        ndc_ray=False,
        is_relight=True,
        N_samples=-1,
    ):

        origins = rays_chunk[:, :3]
        viewdirs = rays_chunk[:, 3:6] # (batch_N, 3)

        print(origins)
        print(viewdirs)

        # if ndc_ray:
        #     xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,
        #                                                          N_samples=N_samples)
        #     dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
        #                       dim=-1)  # dist between 2 consecutive points along a ray
        #     rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        #     dists = dists * rays_norm  # [1, n_sample]
        #     viewdirs = viewdirs / rays_norm
        # else:
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
            t_idx = light_idx[ray_indices]
            if t_origins.shape[0] == 0:
                return torch.zeros(
                    (0, 3), device=t_origins.device
                ), torch.zeros((0,), device=t_origins.device)
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            positions = self.normalize_coord(positions)
            sigmas = self.compute_density(positions)
            radiance_field_feat, intrinsic_feat = self.compute_bothfeature(positions, t_idx)
            rgbs = self.renderModule(positions, t_dirs, radiance_field_feat)

            if is_relight:
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
                # elif self.normals_kind == "residue_prediction":
                #     derived_normals = self.compute_derived_normals(xyz_sampled[app_mask])
                #     predicted_normals = self.renderModule_normal(xyz_sampled[app_mask], derived_normals, intrinsic_feat)
                #     valid_normals = predicted_normals

                #     normals_diff[app_mask] = torch.sum(torch.pow(predicted_normals - derived_normals, 2), dim=-1, keepdim=True)
                #     normals_orientation_loss[app_mask] = torch.sum(viewdirs[app_mask] * predicted_normals, dim=-1, keepdim=True).clamp(min=0)
                self.normal = valid_normals

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

        print('num of rays after sample', ray_indices.shape[0])
        print(ray_indices)
        print(t_starts)

        rgb_map, acc_map, depth_map, extras = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices=ray_indices,
            n_rays=origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            # render_bkgd=torch.ones((3,), device=origins.device),
        )
        print(rgb_map)

        if not is_relight:
            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                rgb_map = rgb_map + (1. - acc_map)

            return  rgb_map, depth_map, None, \
                    None, None, None, \
                    acc_map, None, None, None, \
                    None, None
        else:
            weights = extras['weights']

            normal_map = accumulate_along_rays(
                weights, self.normal, ray_indices, origins.shape[0])
            normals_diff_map = accumulate_along_rays(
                weights, self.normals_diff, ray_indices, origins.shape[0])
            normals_orientation_loss_map = accumulate_along_rays(
                weights, self.normals_orientation_loss, ray_indices, origins.shape[0])
            albedo_map = accumulate_along_rays(
                weights, self.albedo, ray_indices, origins.shape[0])
            roughness_map = accumulate_along_rays(
                weights, self.roughness, ray_indices, origins.shape[0])
            fresnel_map = torch.zeros_like(albedo_map).fill_(self.fixed_fresnel)
            albedo_smoothness_cost_map = accumulate_along_rays(
                weights, self.albedo_smoothness_cost, ray_indices, origins.shape[0])
            roughness_smoothness_cost_map = accumulate_along_rays(
                weights, self.roughness_smoothness_cost, ray_indices, origins.shape[0])

            albedo_smoothness_loss = torch.mean(albedo_smoothness_cost_map)
            roughness_smoothness_loss = torch.mean(roughness_smoothness_cost_map)

            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                rgb_map = rgb_map + (1. - acc_map)
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

            return  rgb_map, depth_map, normal_map, \
                    albedo_map, roughness_map, fresnel_map, \
                    acc_map, normals_diff_map, normals_orientation_loss_map, acc_mask, \
                    albedo_smoothness_loss, roughness_smoothness_loss

    def cal_loss(self, data, args):
        rays = data['rays']
        rgb_gt = data['rgbs']
        light_idx = data['light_idx']
        normal_gt = data['normals']
        device = rays.device

        rgb_map, depth_map, normal_map, albedo_map, roughness_map, \
        fresnel_map, acc_map, normals_diff_map, normals_orientation_loss_map, \
        acc_mask, albedo_smoothness_loss, roughness_smoothness_loss \
        = self.forward(rays, light_idx, is_train=True, white_bg=args.white_bg, is_relight=args.relight_flag, ndc_ray=args.ndc_ray, N_samples=self.nSamples)

        if self.normals_kind == 'gt_normals' and normal_gt is not None:
            normal_map = normal_gt.to(device)

        if args.relight_flag:
            acc_mask = acc_mask.squeeze(-1)
            depth_map = depth_map.squeeze(-1)
            rgb_with_brdf_masked = render_with_BRDF(
                depth_map[acc_mask],
                normal_map[acc_mask],
                albedo_map[acc_mask],
                roughness_map[acc_mask].repeat(1, 3),
                fresnel_map[acc_mask],
                rays[acc_mask],
                self,
                light_idx[acc_mask],
                sample_method=args.light_sample_train,
                chunk_size=args.relight_chunk_size,
                device=device,
                args=args,
            )
            rgb_with_brdf = torch.ones_like(rgb_map) # background default to be white
            rgb_with_brdf[acc_mask] = rgb_with_brdf_masked
            # rgb_with_brdf = rgb_with_brdf * acc_map  + (1. - acc_map)
        else:
            rgb_with_brdf = torch.ones_like(rgb_map)

        loss_dict = {}

        total_loss = 0
        loss_rgb_brdf = torch.tensor(1e-6).to(device)
        loss_rgb = torch.mean((rgb_map - rgb_gt) ** 2)
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
            loss_rgb_brdf = torch.mean((rgb_with_brdf - rgb_gt) ** 2)
            total_loss += loss_rgb_brdf * args.rgb_brdf_weight
            # exponential growth
            normal_weight_factor = args.normals_loss_enhance_ratio ** ((args.nw_iter - args.update_AlphaMask_list[0])/ (args.iteration - args.update_AlphaMask_list[0]))
            BRDF_weight_factor = args.BRDF_loss_enhance_ratio ** ((args.nw_iter - args.update_AlphaMask_list[0])/ (args.iteration - args.update_AlphaMask_list[0]))

            if args.normals_diff_weight > 0:
                loss_normals_diff = normal_weight_factor * args.normals_diff_weight * normals_diff_map.mean()
                total_loss += loss_normals_diff
                loss_dict['loss_normals_diff'] = loss_normals_diff

            if args.normals_orientation_weight > 0:
                loss_normals_orientation = normal_weight_factor * args.normals_orientation_weight * normals_orientation_loss_map.mean()
                total_loss += loss_normals_orientation
                loss_dict['loss_normals_orientation'] = loss_normals_orientation

            if args.roughness_smoothness_loss_weight > 0:
                roughness_smoothness_loss = BRDF_weight_factor * args.roughness_smoothness_loss_weight * roughness_smoothness_loss
                total_loss += roughness_smoothness_loss
                loss_dict['loss_roughness_smoothness'] = roughness_smoothness_loss

            if args.albedo_smoothness_loss_weight > 0:
                albedo_smoothness_loss = BRDF_weight_factor * args.albedo_smoothness_loss_weight * albedo_smoothness_loss
                total_loss += albedo_smoothness_loss
                loss_dict['loss_albedo_smoothness'] = albedo_smoothness_loss

            loss_dict['loss_rgb_brdf'] = loss_rgb_brdf
            loss_dict['PSNR_brdf'] = -10.0 * torch.log10(loss_rgb_brdf)

        loss_dict['total_loss'] = total_loss

        loss_dict['PSNR'] = -10.0 * torch.log10(loss_rgb)
        return loss_dict

    '''
    def evaluation(
        self,
        dataset,
        args,
        device,
        savePath=None,
        N_vis=-1,
        prefix='',
        white_bg=False):
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

        try:
            tqdm._instances.clear()
        except Exception:
            pass

        near_far = dataset.near_far
        W, H = dataset.img_wh

        img_eval_interval = 1 if N_vis < 0 else max(dataset.all_rays.shape[0] // N_vis, 1)
        idxs = list(range(0, dataset.all_rays.shape[0], img_eval_interval))

        _, view_dirs = self.generate_envir_map_dir(256, 512)

        predicted_envir_map = self.get_light_rgbs(view_dirs.reshape(-1, 3).to(device))[0]
        predicted_envir_map = predicted_envir_map.reshape(256, 512, 3).cpu().detach().numpy()
        predicted_envir_map = np.clip(predicted_envir_map, a_min=0, a_max=np.inf)
        predicted_envir_map = np.uint8(np.clip(np.power(predicted_envir_map, 1./2.2), 0., 1.) * 255.)
        envirmap = predicted_envir_map

        # save predicted envir map
        imageio.imwrite(f'{savePath}/envir_map/{prtx}envirmap.png', envirmap)
        test_duration = int(len(test_dataset) / num_test)


        for idx in range(num_test):
            item = test_dataset.__getitem__(idx * test_duration)
            rays = item['rays']                 # [H*W, 6]
            gt_rgb = item['rgbs'][0]            # [H*W, 3]
            light_idx = item['light_idx'][0]    # [H*W, 1]
            gt_rgb_wirh_brdf = gt_rgb           # [H*W, 3]
            gt_mask = item['rgbs_mask']         # [H*W, 1]

            rgb_map, acc_map, depth_map, normal_map, albedo_map, roughness_map, albedo_gamma_map = [], [], [], [], [], [], []
            fresnel_map, rgb_with_brdf_map, normals_diff_map, normals_orientation_loss_map = [], [], [], []

            chunk_idxs = torch.split(torch.arange(rays.shape[0]), args.batch_size_test)
            for chunk_idx in chunk_idxs:
                ret_kw= renderer(
                                    rays[chunk_idx],
                                    None, # not used
                                    light_idx[chunk_idx],
                                    tensoIR,
                                    N_samples=N_samples,
                                    ndc_ray=ndc_ray,
                                    white_bg=white_bg,
                                    sample_method='fixed_envirmap',
                                    chunk_size=args.relight_chunk_size,
                                    device=device,
                                    args=args
                                )
                rgb_map.append(ret_kw['rgb_map'].detach().cpu())
                depth_map.append(ret_kw['depth_map'].detach().cpu())
                normal_map.append(ret_kw['normal_map'].detach().cpu())
                albedo_map.append(ret_kw['albedo_map'].detach().cpu())
                roughness_map.append(ret_kw['roughness_map'].detach().cpu())
                fresnel_map.append(ret_kw['fresnel_map'].detach().cpu())
                rgb_with_brdf_map.append(ret_kw['rgb_with_brdf_map'].detach().cpu())
                normals_diff_map.append(ret_kw['normals_diff_map'].detach().cpu())
                normals_orientation_loss_map.append(ret_kw['normals_orientation_loss_map'].detach().cpu())
                acc_map.append(ret_kw['acc_map'].detach().cpu())


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

            # Store loss and images
            if test_dataset.__len__():
                gt_rgb = gt_rgb.view(H, W, 3)
                gt_rgb_wirh_brdf = gt_rgb_wirh_brdf.view(H, W, 3)
                loss_rgb = torch.mean((rgb_map - gt_rgb) ** 2)
                loss_rgb_brdf = torch.mean((rgb_with_brdf_map - gt_rgb_wirh_brdf) ** 2)
                PSNRs_rgb.append(-10.0 * np.log(loss_rgb.item()) / np.log(10.0))
                PSNRs_rgb_brdf.append(-10.0 * np.log(loss_rgb_brdf.item()) / np.log(10.0))

                if compute_extra_metrics:
                    ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                    l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensoIR.device)
                    l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensoIR.device)

                    ssim_rgb_brdf = rgb_ssim(rgb_with_brdf_map, gt_rgb_wirh_brdf, 1)
                    l_a_rgb_brdf = rgb_lpips(gt_rgb_wirh_brdf.numpy(), rgb_with_brdf_map.numpy(), 'alex', tensoIR.device)
                    l_v_rgb_brdf = rgb_lpips(gt_rgb_wirh_brdf.numpy(), rgb_with_brdf_map.numpy(), 'vgg', tensoIR.device)

                    ssims.append(ssim)
                    l_alex.append(l_a)
                    l_vgg.append(l_v)

                    ssims_rgb_brdf.append(ssim_rgb_brdf)
                    l_alex_rgb_brdf.append(l_a_rgb_brdf)
                    l_vgg_rgb_brdf.append(l_v_rgb_brdf)



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

            if not test_all:
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

                imageio.imwrite(f'{savePath}/nvs_with_radiance_field/{prtx}{idx:03d}.png', rgb_map)
                imageio.imwrite(f'{savePath}/nvs_with_brdf/{prtx}{idx:03d}.png', rgb_with_brdf_map)
                imageio.imwrite(f'{savePath}/normal/{prtx}{idx:03d}.png', normal_map)
                imageio.imwrite(f'{savePath}/normal_vis/{prtx}{idx:03d}.png', normal_rgb_vis_map)
                imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}.png', brdf_map)
                imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}_albedo.png', albedo_gamma_map)
                imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}_roughness.png', roughness_map)
                imageio.imwrite(f'{savePath}/acc_map/{prtx}{idx:03d}.png', acc_map)


        # Randomly select a prediction to visualize
        if logger and step and not test_all:
            vis_idx = random.choice(range(len(rgb_maps)))
            vis_rgb = torch.from_numpy(rgb_maps[vis_idx])
            vis_rgb_brdf_rgb = torch.from_numpy(rgb_with_brdf_maps[vis_idx])
            vis_depth = torch.from_numpy(depth_maps[vis_idx])
            vis_rgb_gt = torch.from_numpy(gt_maps[vis_idx])
            vis_normal_rgb = torch.from_numpy(normal_rgb_maps[vis_idx])
            vis_normals_diff_rgb = torch.from_numpy(normals_diff_maps[vis_idx])
            vis_normals_orientation_loss_rgb = torch.from_numpy(normals_orientation_loss_maps[vis_idx])
            vis_albedo = torch.from_numpy(albedo_maps[vis_idx])
            vis_albedo_gamma = torch.from_numpy(albedo_gamma_maps[vis_idx])
            vis_roughness = torch.from_numpy(roughness_maps[vis_idx])
            vis_fresnel = torch.from_numpy(fresnel_maps[vis_idx])
            vis_rgb_grid = torch.stack([vis_rgb, vis_rgb_brdf_rgb, vis_rgb_gt, vis_depth]).permute(0, 3, 1, 2).to(float)
            vis_normal_grid = torch.stack([vis_normal_rgb, vis_normals_diff_rgb, vis_normals_orientation_loss_rgb]).permute(0, 3, 1, 2).to(float)
            vis_brdf_grid = torch.stack([vis_albedo, vis_roughness, vis_fresnel]).permute(0, 3, 1, 2).to(float)
            vis_envir_map_grid = torch.from_numpy(envirmap).unsqueeze(0).permute(0, 3, 1, 2).to(float)
            vis_albedo_grid = torch.stack([vis_albedo, vis_albedo_gamma]).permute(0, 3, 1, 2).to(float)


            logger.add_image('test/rgb',
                                vutils.make_grid(vis_rgb_grid, padding=0, normalize=True, value_range=(0, 255)), step)
            logger.add_image('test/normal',
                                vutils.make_grid(vis_normal_grid, padding=0, normalize=True, value_range=(0, 255)), step)
            logger.add_image('test/brdf',
                                vutils.make_grid(vis_brdf_grid, padding=0, normalize=True, value_range=(0, 255)), step)
            logger.add_image('test/envir_map',
                                vutils.make_grid(vis_envir_map_grid, padding=0, normalize=True, value_range=(0, 255)), step)
            logger.add_image('test/albedo',
                                vutils.make_grid(vis_albedo_grid, padding=0, normalize=True, value_range=(0, 255)), step)


        # Compute metrics
        if PSNRs_rgb:
            psnr = np.mean(np.asarray(PSNRs_rgb))
            psnr_rgb_brdf = np.mean(np.asarray(PSNRs_rgb_brdf))


            if compute_extra_metrics:
                ssim = np.mean(np.asarray(ssims))
                l_a = np.mean(np.asarray(l_alex))
                l_v = np.mean(np.asarray(l_vgg))

                ssim_rgb_brdf = np.mean(np.asarray(ssims_rgb_brdf))
                l_a_rgb_brdf = np.mean(np.asarray(l_alex_rgb_brdf))
                l_v_rgb_brdf = np.mean(np.asarray(l_vgg_rgb_brdf))



                saved_message = f'Iteration:{prtx[:-1]}: \n' \
                                + f'\tPSNR_nvs: {psnr:.2f}, PSNR_nvs_brdf: {psnr_rgb_brdf:.2f}\n' \
                                + f'\tSSIM_rgb: {ssim:.4f}, L_Alex_rgb: {l_a:.4f}, L_VGG_rgb: {l_v:.4f}\n' \
                                + f'\tSSIM_rgb_brdf: {ssim_rgb_brdf:.4f}, L_Alex_rgb_brdf: {l_a_rgb_brdf:.4f}, L_VGG_rgb_brdf: {l_v_rgb_brdf:.4f}\n'

            else:
                saved_message = f'Iteration:{prtx[:-1]}, PSNR_nvs: {psnr:.2f}, PSNR_nvs_brdf: {psnr_rgb_brdf:.2f}\n'
            # write the end of record file
            with open(f'{savePath}/metrics_record.txt', 'a') as f:
                f.write(saved_message)

        return psnr, psnr_rgb_brdf
    '''