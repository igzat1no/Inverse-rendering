import numpy as np
import torch
import torch.nn as nn

from models.myutils import scale_anything
from models.neus import NeuS
from render import mesh
from render import render
from render import regularizer

from geometry import dmtet
from render import obj
from geometry import utils

class NeuS_DMTet(nn.Module):
    def __init__(self, sdf_net, device, conf):
        super(NeuS_DMTet, self).__init__()

        self.sdf_net = sdf_net.geometry
        self.device = device
        self.conf = conf
        self.marching_tets = dmtet.DMTet()
        self.grid_res = conf.dmtet.dmtet_grid
        self.scale = conf.dmtet.mesh_scale

        if conf.unbounded:
            tets = np.load('data/tets/sphere_{}_tets.npz'.format(self.grid_res))
        else:
            tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))

        self.verts = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * self.scale
        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.density_threshold = torch.nn.Parameter(torch.tensor(-3.0), requires_grad=True)

        print("verts", self.verts.shape)
        print("indices", self.indices.shape)

        # self.generate_edges()

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)

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
        sdf, _ = self.sdf_net(v_deformed, with_grad=False)
        # v_deformed = self.verts.detach()
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, sdf, self.indices)  # 14850M
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh

    def render(self, glctx, target, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        # obj.write_ply(".", opt_mesh)
        # input()
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], target['resolution'], spp=target['spp'],
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
        buffers = self.render(glctx, target, opt_material)  # 15949M

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']

        # from PIL import Image
        # nw = color_ref.squeeze(dim=0)
        # nw = nw.cpu().detach().numpy()
        # im = Image.fromarray(np.uint8(nw*255))
        # im.save("ref.png")

        # nw = buffers['shaded'].squeeze(dim=0)
        # nw = nw.cpu().detach().numpy()
        # im = Image.fromarray(np.uint8(nw*255))
        # im.save("my.png")
        # input()

        if color_ref.shape[-1] == 4:
            img_loss = loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
            img_loss = img_loss + torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        else:
            img_loss = loss_fn(buffers['shaded'][..., 0:3], color_ref[..., 0:3])

        points_ = buffers["gb_pos"].view(-1, 3)
        points = scale_anything(points_, (-self.scale, self.scale), (0, 1)) # points normalized to (0, 1)

        sdf, _ = self.sdf_net(points, with_grad=False)
        grad = torch.autograd.grad(
            sdf, points, grad_outputs=torch.ones_like(sdf),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        loss_eikonal = ((torch.linalg.norm(grad, ord=2, dim=-1) - 1.)**2).mean()
        reg_loss = loss_eikonal * self.conf.sdf_grad_weight

        loss_sparsity = torch.exp(-1 * sdf.abs()).mean()
        reg_loss += loss_sparsity * self.conf.sdf_sparse_weight

        return img_loss, reg_loss # torch.Tensor([0.0]).cuda()