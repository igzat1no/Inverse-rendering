import mcubes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.myutils import scale_anything
from models.network_utils import get_encoding, get_mlp


class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()

        self.config = config
        self.n_dir_dims = config.get('n_dir_dims', 3)
        self.n_output_dims = 3

        self.encoding = get_encoding(self.n_dir_dims, config.dir_encoding_config)
        self.n_input_dims = config.input_feature_dim + self.encoding.n_output_dims
        self.network = get_mlp(self.n_input_dims, self.n_output_dims, config.mlp_network_config)

    def forward(self, features, dirs, *args):
        dirs = (dirs + 1.) / 2.  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        # feature + dirs_embed + normal
        # 13 + 16 + 3 = 32
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + \
                                [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        color = torch.sigmoid(color)  # (0, 1)
        return color

    def update_step(self, epoch, global_step):
        if hasattr(self.encoding, 'update_step'):
            self.encoding.update_step(epoch, global_step)


class MarchingCubeHelper(nn.Module):
    def __init__(self, resolution, device="cuda"):
        super().__init__()
        self.resolution = resolution
        self.device = device
        # self.points_range = (-radius, radius)
        self.points_range = (0, 1)
        self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z, indexing='ij')
            verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def forward(self, level, threshold=0.):
        level = level.float().view(self.resolution, self.resolution, self.resolution)
        verts, faces = self.mc_func(-level.numpy(), threshold) # transform to numpy
        verts = torch.from_numpy(verts.astype(np.float32))
        faces = torch.from_numpy(faces.astype(np.int64)) # transform back to pytorch
        verts = verts / (self.resolution - 1.)
        return {
            'v_pos': verts,
            't_pos_idx': faces
        }


class VolumeSDF(nn.Module):
    def __init__(self, config, device):
        super(VolumeSDF, self).__init__()

        self.config = config
        self.device = device
        self.n_output_dims = config.feature_dim
        self.radius = config.radius

        self.encoding = get_encoding(3, config.xyz_encoding_config)
        self.network = get_mlp(self.encoding.n_output_dims, self.n_output_dims,
                               config.mlp_network_config)
        if config.isosurface is not None:
            self.helper = MarchingCubeHelper(config.isosurface.resolution,
                                             device=self.device)

    def forward(self, points, with_grad=False):
        with torch.set_grad_enabled(self.training or with_grad):
            if with_grad:
                if not self.training:
                    points = points.clone()  # points may be in inference mode, get a copy to enable grad
                points.requires_grad_(True)
            points_ = points
            points = scale_anything(points_, (-self.radius, self.radius), (0, 1))
            out = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
            sdf, feature = out[...,0], out
            if with_grad:  # need to compute normal for points
                grad = torch.autograd.grad(
                    sdf, points_, grad_outputs=torch.ones_like(sdf),
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]

        rv = [sdf]
        rv.append(feature)
        if with_grad:
            rv.append(grad)  # type: ignore
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points):
        points = scale_anything(points, (-self.radius, self.radius), (0, 1))
        sdf = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        return sdf

    def isosurface_(self, vmin, vmax):
        vertices = self.helper.grid_vertices()
        chunk_size = self.config.isosurface.chunk
        num_vertices = vertices.shape[0]

        level = []
        for i in range(0, num_vertices, chunk_size):
            nw_vertices = vertices[i:i+chunk_size]
            nw_vertices = torch.stack([
                scale_anything(nw_vertices[...,0], (0, 1), (vmin[0], vmax[0])),
                scale_anything(nw_vertices[...,1], (0, 1), (vmin[1], vmax[1])),
                scale_anything(nw_vertices[...,2], (0, 1), (vmin[2], vmax[2])),
            ], dim=-1).to(self.device)
            ret = self.forward_level(nw_vertices).cpu()
            level.append(ret)
        level = torch.cat(level, dim=0)

        mesh = self.helper(level, threshold=self.config.isosurface.threshold)
        mesh['v_pos'] = torch.stack([
            scale_anything(mesh['v_pos'][...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(mesh['v_pos'][...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(mesh['v_pos'][...,2], (0, 1), (vmin[2], vmax[2]))
        ], dim=-1)
        return mesh

    @torch.no_grad()
    def isosurface(self):
        mesh_coarse = self.isosurface_((-self.radius, -self.radius, -self.radius), (self.radius, self.radius, self.radius))
        vmin, vmax = mesh_coarse['v_pos'].amin(dim=0), mesh_coarse['v_pos'].amax(dim=0)
        vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        mesh_fine = self.isosurface_(vmin_, vmax_)
        return mesh_fine
