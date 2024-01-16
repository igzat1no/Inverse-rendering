# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

from render import mesh
from render import render
from render import regularizer

from tensorf.utils import convert_sdf_samples_to_mesh

import trimesh
import numpy as np

from geometry import dmtet
from render import obj

###############################################################################
#  Geometry interface
###############################################################################

class DLMesh(torch.nn.Module):
    def __init__(self, initial_guess, FLAGS):
        super(DLMesh, self).__init__()

        self.FLAGS = FLAGS

        self.initial_guess = initial_guess
        self.mesh          = initial_guess.clone()
        print("Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))
        
        self.mesh.v_pos = torch.nn.Parameter(self.mesh.v_pos, requires_grad=True)
        self.register_parameter('vertex_pos', self.mesh.v_pos)

        # self.deform = torch.nn.Parameter(torch.zeros_like(self.mesh.v_pos), requires_grad=True)
        # self.register_parameter('deform', self.deform)

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material):

        # self.mesh.v_pos = self.initial_guess.v_pos + 1/self.FLAGS.dmtet_grid*torch.tanh(self.deform)
        self.mesh.material = material

        imesh = mesh.Mesh(base=self.mesh)
        # # Compute normals and tangent space
        # imesh = mesh.auto_normals(imesh)
        # imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                    num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf, tex_type=self.FLAGS.tex_type,
                                    downsample = self.FLAGS.downsample, anti_aliasing = self.FLAGS.anti_aliasing, anti_aliasing_mode = self.FLAGS.anti_aliasing_mode)

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

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Compute regularizer. 
        if self.FLAGS.laplace == "absolute":
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos, self.mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)
        elif self.FLAGS.laplace == "relative":
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos - self.initial_guess.v_pos, self.mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)                

        # # Albedo (k_d) smoothnesss regularizer
        # reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # # Visibility regularizer
        # reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # # Light white balance regularizer
        # reg_loss = reg_loss + lgt.regularizer() * 0.005

        return img_loss, reg_loss

class TensoRF_MCGeometry(torch.nn.Module):
    def __init__(self, sdf_net, FLAGS):
        super(TensoRF_MCGeometry, self).__init__()

        self.FLAGS = FLAGS
        self.sdf_net = sdf_net
        self.marching_tets = dmtet.DMTet()

        tets = np.load('data/tets/{}_tets.npz'.format(self.FLAGS.dmtet_grid))
        self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * sdf_net.aabb.abs().max()*2
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        sdf = self.sdf_net.compute_alpha(self.verts, self.sdf_net.stepSize)-FLAGS.sdf_level
        vertices, faces, uvs, uv_idx = self.marching_tets(self.verts, sdf, self.indices)

        self.mesh = mesh.Mesh(vertices, faces)
        obj.write_ply("./", self.mesh, filename="test1_mesh.ply")

        if self.FLAGS.unbounded:
            print("unwarp")
            import ipdb; ipdb.set_trace()
            pts_norm = torch.norm(vertices, dim=-1)
            scale = 1/(self.sdf_net.aabb[1][0]-pts_norm[..., None]) / pts_norm[..., None]
            mask_inside_inner_sphere = (pts_norm <= 1.0)[..., None]
            vertices = torch.where(mask_inside_inner_sphere, vertices, scale * vertices)

        # in_mesh = trimesh.load(FLAGS.base_mesh)
        # vertices = torch.from_numpy(in_mesh.vertices).float().cuda()
        # faces = torch.from_numpy(in_mesh.faces).long().cuda()


        print("vertices: ", vertices.shape)
        print("faces: ", faces.shape)
        self.mesh = mesh.Mesh(vertices, faces)

        obj.write_ply("./", self.mesh, filename="test2_mesh.ply")

        if self.FLAGS.unbounded:
            print("warp")
            pts_norm = torch.norm(vertices, dim=-1)
            scale = (self.sdf_net.aabb[1][0] - 1.0 / pts_norm[..., None]) / pts_norm[..., None]
            mask_inside_inner_sphere = (pts_norm <= 1.0)[..., None]
            vertices = torch.where(mask_inside_inner_sphere, vertices, scale * vertices)

        self.mesh = mesh.Mesh(vertices, faces)
        obj.write_ply("./", self.mesh, filename="test3_mesh.ply")

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material):
        self.mesh.material = material

        imesh = mesh.Mesh(base=self.mesh)
        # # Compute normals and tangent space
        # imesh = mesh.auto_normals(imesh)
        # imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                    num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf, tex_type=self.FLAGS.tex_type,
                                    downsample = self.FLAGS.downsample, anti_aliasing = self.FLAGS.anti_aliasing, anti_aliasing_mode = self.FLAGS.anti_aliasing_mode)

class NeuS_MCGeometry(torch.nn.Module):
    def __init__(self, sdf_net, FLAGS):
        super(NeuS_MCGeometry, self).__init__()
        self.FLAGS = FLAGS

        in_mesh = trimesh.load(FLAGS.base_mesh)

        vertices = torch.from_numpy(in_mesh.vertices).float().cuda()
        faces = torch.from_numpy(in_mesh.faces).long().cuda()
        print("vertices: ", vertices.shape)
        print("faces: ", faces.shape)
        self.mesh = mesh.Mesh(vertices, faces)

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material):
        self.mesh.material = material

        imesh = mesh.Mesh(base=self.mesh)
        # # Compute normals and tangent space
        # imesh = mesh.auto_normals(imesh)
        # imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                    num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf, tex_type=self.FLAGS.tex_type,
                                    downsample = self.FLAGS.downsample, anti_aliasing = self.FLAGS.anti_aliasing, anti_aliasing_mode = self.FLAGS.anti_aliasing_mode)