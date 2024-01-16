import numpy as np
import torch
import torch.nn as nn

from geometry import dmtet
from render import mesh, regularizer, render, obj

from models import sampler


class Basemodel_dmtet(nn.Module):
    def __init__(self, sdf_net, device, conf, init_scale=1.):
        super().__init__()

        self.sdf_net = sdf_net
        self.device = device
        self.conf = conf
        self.init_scale = init_scale

        self.marching_tets = dmtet.DMTet()
        self.grid_res = conf.dmtet.dmtet_grid
        self.scale = conf.dmtet.mesh_scale
        if conf.unbounded:
            tets = np.load('data/tets/sphere_{}_tets.npz'.format(self.grid_res))
        else:
            tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        self.verts = torch.tensor(tets['vertices'], dtype=torch.float32, device=device) * self.scale
        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device=device)
        self.density_threshold = torch.nn.Parameter(torch.tensor(-3.0), requires_grad=True)

        self.deform = nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    @torch.no_grad()
    def update_scale(self, new_scale):
        self.verts = self.verts / self.scale * new_scale
        self.scale = new_scale

    def get_density_threshold(self):
        return 0.5 * torch.tanh(self.density_threshold) + 0.502

    def cal_sdf(self, v):
        raise NotImplementedError("Please implement cal_sdf function")

    def getMesh(self, material):
        # Run DM tet to get a base mesh
        v_deformed = self.verts + 1 / self.grid_res * torch.tanh(self.deform)
        sdf = self.cal_sdf(v_deformed)
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, sdf, self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, opt_material, bsdf=None):
        self.opt_mesh = self.getMesh(opt_material)

        # visualize mesh using trimesh
        # obj.write_ply(".", self.opt_mesh, save_material=False)

        return render.render_mesh(glctx,
                                  self.opt_mesh,
                                  target['mvp'],
                                  target['campos'],
                                  target['resolution'],
                                  spp=target['spp'],
                                  msaa=True,
                                  background=target['background'],
                                  bsdf=bsdf,
                                  tex_type=self.conf.tex_type,
                                  downsample=self.conf.downsample,
                                  anti_aliasing=self.conf.anti_aliasing,
                                  anti_aliasing_mode=self.conf.anti_aliasing_mode)

    def tick(self, glctx, target, opt_material, loss_fn, iteration):
        buffers = self.render(glctx, target, opt_material)
        return self.cal_loss(buffers, target, loss_fn, iteration, opt_material)

    def cal_loss(self, buffers, target, loss_fn, iteration, mat=None):
        raise NotImplementedError("Please implement cal_loss function")