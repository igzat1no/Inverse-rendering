# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import torch

from render import mesh
from render import render
from render import regularizer

from geometry import dmtet
from render import obj
from geometry import utils

class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

###############################################################################
#  Geometry interface
###############################################################################

class SDF_DMTetGeometry(torch.nn.Module):
    def __init__(self, sdf_net, grid_res, scale, FLAGS, init_scale=1.):
        super(SDF_DMTetGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.marching_tets = dmtet.DMTet()
        self.sdf_net       = sdf_net
        self.scale         = scale
        self.init_scale    = init_scale
        if FLAGS.unbounded:
            # tets = np.load('data/tets/sphere_{}_tets.npz'.format(self.grid_res))
            tets = np.load('data/tets/mix_256_{}_tets.npz'.format(self.grid_res))
            scale = 3.98
        else:
            tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.density_threshold = torch.nn.Parameter(torch.tensor(FLAGS.density_threshold_init), requires_grad=True)

        # from tqdm import tqdm
        # file = open("a.obj", 'w')
        # for i in tqdm(range(self.verts.shape[0])):
        #     file.write("v {} {} {}\n".format(self.verts[i][0].item(), self.verts[i][1].item(), self.verts[i][2].item()))
        # file.close()

        print("verts", self.verts.shape)
        print("indices", self.indices.shape)

        # import ipdb; ipdb.set_trace()

        # self.generate_edges()

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            print("edges", edges.shape, edges.nelement() * edges.element_size()/1024/1024/1024)
            all_edges = self.indices[:,edges].reshape(-1,2)
            print("all_edges", all_edges.shape, all_edges.nelement() * all_edges.element_size()/1024/1024/1024)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            print("all_edges_sorted", all_edges_sorted.shape, all_edges_sorted.nelement() * all_edges_sorted.element_size()/1024/1024/1024)
            self.all_edges = torch.unique(all_edges_sorted, dim=0)
            print("self.all_edges", self.all_edges.shape, self.all_edges.nelement() * self.all_edges.element_size()/1024/1024/1024)

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    @torch.no_grad()
    def update_scale(self, new_scale):
        self.verts = self.verts / self.scale * new_scale
        self.scale = new_scale

    def get_density_threshold(self):
        return 0.5*torch.tanh(self.density_threshold)+0.501
        # return 0.005

    def getMesh(self, material):
        # Run DM tet to get a base mesh
        v_deformed = self.verts + 1 / self.grid_res * torch.tanh(self.deform)
        sdf = self.sdf_net.compute_alpha(v_deformed, self.sdf_net.stepSize)*self.init_scale-self.get_density_threshold()
        # v_deformed = self.verts.detach()
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, sdf, self.indices)

        if self.FLAGS.unbounded:
            verts = utils.unbounded_unwarp(verts, radius=self.sdf_net.aabb[1][0])
            # pts_norm = torch.norm(verts, dim=-1)
            # scale = 1/(self.sdf_net.aabb[1][0]-pts_norm[..., None]) / pts_norm[..., None]
            # mask_inside_inner_sphere = (pts_norm <= 1.0)[..., None]
            # verts = torch.where(mask_inside_inner_sphere, verts, scale * verts)


        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # obj.write_ply("./", imesh, filename="test_mesh.ply")
        # import ipdb; ipdb.set_trace()

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        self.opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, self.opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'],
                                        msaa=True, background=target['background'], bsdf=bsdf, tex_type=self.FLAGS.tex_type,
                                        downsample = self.FLAGS.downsample, anti_aliasing = self.FLAGS.anti_aliasing, anti_aliasing_mode = self.FLAGS.anti_aliasing_mode)

    def run_tick(self, glctx, lgt, loss_fn, iteration):
        def func(target, opt_material):
            return self.tick(glctx, target, lgt, opt_material, loss_fn, iteration)
        return func

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        if color_ref.shape[-1] == 4:
            img_loss = loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
            img_loss = img_loss + torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        else:
            img_loss = loss_fn(buffers['shaded'][..., 0:3], color_ref[..., 0:3])


        if abs(self.FLAGS.normal_smooth_weight) > 1e-4:
            reg_loss = regularizer.normal_consistency(self.opt_mesh.v_pos, self.opt_mesh.t_pos_idx) * self.FLAGS.normal_smooth_weight
        else:
            reg_loss = torch.Tensor([0.0]).cuda()

        # tvreg = TVLoss()
        # loss_tv = self.sdf_net.TV_loss_density(tvreg)
        # reg_loss = reg_loss + loss_tv
        # loss_tv = self.sdf_net.TV_loss_app(tvreg)
        # reg_loss = reg_loss + loss_tv

        # # Neus SDF regularizer
        # points_ = buffers["gb_pos"].view(-1, 3)
        # eps = 0.003
        # points_d_ = torch.stack([
        #     points_ + torch.as_tensor([eps, 0.0, 0.0]).to(points_),
        #     points_ + torch.as_tensor([-eps, 0.0, 0.0]).to(points_),
        #     points_ + torch.as_tensor([0.0, eps, 0.0]).to(points_),
        #     points_ + torch.as_tensor([0.0, -eps, 0.0]).to(points_),
        #     points_ + torch.as_tensor([0.0, 0.0, eps]).to(points_),
        #     points_ + torch.as_tensor([0.0, 0.0, -eps]).to(points_)
        # ], dim=0)
        # points_d_sdf = self.sdf_net.compute_alpha(points_d_.view(-1,3), self.sdf_net.stepSize).view(6, -1).float()-self.get_density_threshold()
        # grad = torch.stack([
        #     0.5 * (points_d_sdf[0] - points_d_sdf[1]) / eps,
        #     0.5 * (points_d_sdf[2] - points_d_sdf[3]) / eps,
        #     0.5 * (points_d_sdf[4] - points_d_sdf[5]) / eps,
        # ], dim=-1)

        # loss_eikonal = ((torch.linalg.norm(grad, ord=2, dim=-1) - 1.)**2).mean()
        # reg_loss = loss_eikonal * self.FLAGS.sdf_grad_weight

        # loss_sparsity = torch.exp(-1 * points_d_sdf.abs()).mean()
        # reg_loss += loss_sparsity * self.FLAGS.sdf_sparse_weight

        # # SDF regularizer
        # sdf = self.sdf_net.compute_alpha(self.verts, self.sdf_net.stepSize)*self.init_scale-self.get_density_threshold()
        # sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01)*min(1.0, 4.0 * t_iter)
        # reg_loss = dmtet.sdf_reg_loss(sdf, self.all_edges).mean() * sdf_weight # Dropoff to 0.01

        # # Albedo (k_d) smoothnesss regularizer
        # reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # # Visibility regularizer
        # reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # # Light white balance regularizer
        # reg_loss = reg_loss + lgt.regularizer() * 0.005

        return img_loss, reg_loss

from neus.utils import scale_anything
class NeuS_DMTetGeometry(torch.nn.Module):
    def __init__(self, sdf_net, grid_res, scale, FLAGS, init_scale=1.):
        super(NeuS_DMTetGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.marching_tets = dmtet.DMTet()
        self.sdf_net       = sdf_net
        self.scale         = scale
        self.init_scale    = init_scale

        if FLAGS.unbounded:
            tets = np.load('data/tets/sphere_{}_tets.npz'.format(self.grid_res))
        else:
            tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.density_threshold = torch.nn.Parameter(torch.tensor(-3.0), requires_grad=True)

        print("verts", self.verts.shape)
        print("indices", self.indices.shape)

        # self.generate_edges()

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            print("edges", edges.shape, edges.nelement() * edges.element_size()/1024/1024/1024)
            all_edges = self.indices[:,edges].reshape(-1,2)
            print("all_edges", all_edges.shape, all_edges.nelement() * all_edges.element_size()/1024/1024/1024)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            print("all_edges_sorted", all_edges_sorted.shape, all_edges_sorted.nelement() * all_edges_sorted.element_size()/1024/1024/1024)
            self.all_edges = torch.unique(all_edges_sorted, dim=0)
            print("self.all_edges", self.all_edges.shape, self.all_edges.nelement() * self.all_edges.element_size()/1024/1024/1024)

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    @torch.no_grad()
    def update_scale(self, new_scale):
        self.verts = self.verts / self.scale * new_scale
        self.scale = new_scale

    def get_density_threshold(self):
        return 0.5*torch.tanh(self.density_threshold)+0.502
        # return 0.005

    def getMesh(self, material):
        # Run DM tet to get a base mesh
        v_deformed = self.verts + 1 / self.grid_res * torch.tanh(self.deform)
        sdf = self.sdf_net.geometry(v_deformed, with_grad=False, with_feature=False)
        # v_deformed = self.verts.detach()
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, sdf, self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'],
                                        msaa=True, background=target['background'], bsdf=bsdf, tex_type=self.FLAGS.tex_type,
                                        downsample = self.FLAGS.downsample, anti_aliasing = self.FLAGS.anti_aliasing, anti_aliasing_mode = self.FLAGS.anti_aliasing_mode)

    def run_tick(self, glctx, lgt, loss_fn, iteration):
        def func(target, opt_material):
            return self.tick(glctx, target, lgt, opt_material, loss_fn, iteration)
        return func

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        if color_ref.shape[-1] == 4:
            img_loss = loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
            img_loss = img_loss + torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        else:
            img_loss = loss_fn(buffers['shaded'][..., 0:3], color_ref[..., 0:3])

        points_ = buffers["gb_pos"].view(-1, 3)
        points = scale_anything(points_, (-self.scale, self.scale), (0, 1)) # points normalized to (0, 1)

        sdf = self.sdf_net.geometry(points, with_grad=False, with_feature=False)
        grad = torch.autograd.grad(
            sdf, points, grad_outputs=torch.ones_like(sdf),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        loss_eikonal = ((torch.linalg.norm(grad, ord=2, dim=-1) - 1.)**2).mean()
        reg_loss = loss_eikonal * self.FLAGS.sdf_grad_weight

        loss_sparsity = torch.exp(-1 * sdf.abs()).mean()
        reg_loss += loss_sparsity * self.FLAGS.sdf_sparse_weight

        return img_loss, reg_loss # torch.Tensor([0.0]).cuda()