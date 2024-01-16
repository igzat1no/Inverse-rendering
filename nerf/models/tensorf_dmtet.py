import numpy as np
import torch
import torch.nn as nn

from models.basemodel_dmtet import Basemodel_dmtet
from models.tensoRF import TensorVM, TensorCP
from geometry import dmtet
from render import mesh, regularizer, render
from render import obj


class TensorVM_DMTet_old(nn.Module):
    def __init__(self, sdf_net, device, conf, init_scale=1.):
        super(TensorVM_DMTet, self).__init__()

        self.sdf_net = sdf_net
        self.device = device
        self.conf = conf
        self.marching_tets = dmtet.DMTet()
        self.grid_res = conf.dmtet.dmtet_grid
        self.scale = conf.dmtet.mesh_scale
        self.init_scale = 1.

        if conf.unbounded:
            tets = np.load('data/tets/mix_256_{}_tets.npz'.format(self.grid_res))
        else:
            tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))

        self.verts = torch.tensor(tets['vertices'], dtype=torch.float32, device=device) * self.scale  # here we do not consider unbounded scenes, otherwise we need change scale
        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device=device)
        self.density_threshold = nn.Parameter(torch.tensor(-3.0), requires_grad=True)

        print("device", self.device)
        print("verts", self.verts.shape)
        print("indices", self.indices.shape)

        # self.generate_edges()

        self.deform = nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
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
        sdf = self.sdf_net.compute_alpha(v_deformed, self.sdf_net.render_step_size) * self.init_scale - self.get_density_threshold()
        # v_deformed = self.verts.detach()
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, sdf, self.indices)

        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # obj.write_ply("./", imesh, filename="test_mesh.ply")
        # import ipdb; ipdb.set_trace()

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh

    def render(self, glctx, target, opt_material, bsdf=None):
        self.opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, self.opt_mesh, target['mvp'], target['campos'],  target['resolution'], spp=target['spp'],
                                        msaa=True, background=target['background'], bsdf=bsdf, tex_type=self.conf.tex_type,
                                        downsample = self.conf.downsample, anti_aliasing = self.conf.anti_aliasing, anti_aliasing_mode = self.conf.anti_aliasing_mode)

    def run_tick(self, glctx, lgt, loss_fn, iteration):
        def func(target, opt_material):
            return self.tick(glctx, target, opt_material, loss_fn, iteration)
        return func

    def tick(self, glctx, target, opt_material, loss_fn, iteration):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']

        if color_ref.shape[-1] == 4:
            img_loss = loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
            img_loss = img_loss + torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        else:
            img_loss = loss_fn(buffers['shaded'][..., 0:3], color_ref[..., 0:3])


        if abs(self.conf.normal_smooth_weight) > 1e-4:
            reg_loss = regularizer.normal_consistency(self.opt_mesh.v_pos, self.opt_mesh.t_pos_idx) * self.conf.normal_smooth_weight
        else:
            reg_loss = torch.Tensor([0.0]).cuda()

        return img_loss, reg_loss


class TensorVM_DMTet(Basemodel_dmtet):
    def cal_sdf(self, v):
        sdf = self.sdf_net.compute_alpha(v, self.sdf_net.render_step_size) * self.init_scale - \
            self.get_density_threshold()
        return sdf

    def cal_loss(self, buffers, target, loss_fn, iteration):
        color_ref = target['img']
        if color_ref.shape[-1] == 4:
            img_loss = loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
            img_loss = img_loss + torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        else:
            img_loss = loss_fn(buffers['shaded'][..., 0:3], color_ref[..., 0:3])


        if abs(self.conf.normal_smooth_weight) > 1e-4:
            reg_loss = regularizer.normal_consistency(self.opt_mesh.v_pos, self.opt_mesh.t_pos_idx) * self.conf.normal_smooth_weight
        else:
            reg_loss = torch.Tensor([0.0]).cuda()

        return img_loss, reg_loss