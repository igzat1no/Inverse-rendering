import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tensoIR.relight_utils import grid_sample


class CPModule(nn.Module):
    '''Factorize the model with CP decomposition.
    '''
    def __init__(self, n_comp, gridSize, scale=0.2, dim=3):
        super(CPModule, self).__init__()
        self.n_comp = n_comp
        self.gridSize = gridSize
        self.scale = scale
        self.dim = dim

        self.param = []
        for i in range(dim):
            self.param.append(
                nn.Parameter(scale * torch.randn(1, n_comp, gridSize[i], 1)))
        self.param = nn.ParameterList(self.param)

    def compute_feature(self, xyz_sampled):
        coordinate_line = torch.stack(
            [xyz_sampled[:, i] for i in range(self.dim)])
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line),
            dim=-1).detach().view(3, -1, 1, 2)

        line_coef_point = F.grid_sample(
            self.param[0], coordinate_line[[0]], align_corners=True).view(
                -1, *xyz_sampled.shape[:1])
        for i in range(1, self.dim):
            line_coef_point = line_coef_point * F.grid_sample(
                self.param[i], coordinate_line[[i]], align_corners=True).view(
                    -1, *xyz_sampled.shape[:1])

        return line_coef_point

    @torch.no_grad()
    def upsample(self, res_target):
        for i in range(self.dim):
            self.param[i] = nn.Parameter(
                F.interpolate(self.param[i].data,
                              size=(res_target[i], 1),
                              mode="bilinear",
                              align_corners=True))

    @torch.no_grad()
    def shrink(self, l, r):
        for i in range(self.dim):
            self.param[i] = nn.Parameter(
                self.param[i].data[:, :, l[i]:r[i], :])

    def L1_loss(self):
        loss = 0
        for i in range(self.dim):
            loss += torch.mean(torch.abs(self.param[i]))
        return loss

    def TV_loss(self, reg):
        loss = 0
        for i in range(self.dim):
            loss += reg(self.param[i]) * 1e-3
        return loss


class DensityLine(CPModule):
    def __init__(self, n_comp, gridSize, scale=0.2, dim=3):
        super(DensityLine, self).__init__(n_comp, gridSize, scale, dim)

    def compute(self, xyz_sampled):
        line_coef_point = self.compute_feature(xyz_sampled)
        return torch.sum(line_coef_point, dim=0)


class AppLine(CPModule):
    def __init__(self, n_comp, gridSize, app_dim, scale=0.2, dim=3):
        super(AppLine, self).__init__(n_comp, gridSize, scale, dim)
        self.app_dim = app_dim
        self.mat = nn.Linear(n_comp, app_dim, bias=False)

    def compute(self, xyz_sampled):
        line_coef_point = self.compute_feature(xyz_sampled)
        return self.mat(line_coef_point.T)


class VMModule(nn.Module):
    '''Plane + line, for VM decomposition.
    '''
    def __init__(self, n_comp, gridSize, scale=0.1, dim=3):
        super(VMModule, self).__init__()
        self.n_comp = n_comp
        self.gridSize = gridSize
        self.scale = scale
        self.dim = dim

        self.matMode = []
        for i in range(dim):
            nw = []
            for j in range(dim):
                if j != i:
                    nw.append(j)
            self.matMode.append(nw)

        self.plane, self.line = [], []
        # now we only implement for dim=3
        for i in range(self.dim):
            id_0, id_1 = self.matMode[i]
            self.plane.append(
                nn.Parameter(scale * torch.randn(1, n_comp[i], gridSize[id_1], gridSize[id_0])))
            self.line.append(
                nn.Parameter(scale * torch.randn(1, n_comp[i], gridSize[i], 1)))
        self.plane = nn.ParameterList(self.plane)
        self.line = nn.ParameterList(self.line)

    def compute_feature(self, xyz_sampled):
        coordinate_plane = torch.stack([xyz_sampled[..., self.matMode[i]]
            for i in range(self.dim)]).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack([xyz_sampled[:, i] for i in range(self.dim)])
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line),
            dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for i in range(self.dim):
            plane_coef_point.append(F.grid_sample(self.plane[i], coordinate_plane[[i]],
                        align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.line[i], coordinate_line[[i]],
                        align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point = torch.cat(plane_coef_point)
        line_coef_point = torch.cat(line_coef_point)

        return plane_coef_point * line_coef_point

    def compute_feature_with_grad(self, xyz_sampled):
        coordinate_plane = torch.stack([xyz_sampled[..., self.matMode[i]]
            for i in range(self.dim)]).view(3, -1, 1, 2)
        coordinate_line = torch.stack([xyz_sampled[:, i] for i in range(self.dim)])
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line),
            dim=-1).view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for i in range(self.dim):
            plane_coef_point.append(grid_sample(self.plane[i],
                coordinate_plane[[i]]).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(grid_sample(self.line[i],
                coordinate_line[[i]]).view(-1, *xyz_sampled.shape[:1]))

        plane_coef_point = torch.cat(plane_coef_point)
        line_coef_point = torch.cat(line_coef_point)
        return plane_coef_point * line_coef_point

    def vectorDiff(self):
        total = 0
        for i in range(self.dim):
            n_comp, n_size = self.line[i].shape[1:-1]
            dotp = torch.matmul(self.line[i].view(n_comp, n_size),
                                self.line[i].view(n_comp, n_size).transpose(-1, -2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total += torch.mean(torch.abs(non_diagonal))
        return total

    @torch.no_grad()
    def upsample(self, res_target):
        for i in range(self.dim):
            id_0, id_1 = self.matMode[i]
            self.plane[i] = nn.Parameter(
                F.interpolate(self.plane[i].data,
                              size=(res_target[id_1], res_target[id_0]),
                              mode='bilinear',
                              align_corners=True))
            self.line[i] = nn.Parameter(
                F.interpolate(self.line[i].data,
                              size=(res_target[i], 1),
                              mode='bilinear',
                              align_corners=True))

    @torch.no_grad()
    def shrink(self, t_l, b_r):
        for i in range(self.dim):
            self.line[i] = nn.Parameter(
                self.line[i].data[..., t_l[i]:b_r[i], :])
            id_0, id_1 = self.matMode[i]
            self.plane[i] = nn.Parameter(
                self.plane[i].data[..., t_l[id_1]:b_r[id_1], t_l[id_0]:b_r[id_0]])

    def L1_loss(self):
        loss = 0
        for i in range(self.dim):
            loss += torch.mean(torch.abs(self.plane[i]))
            loss += torch.mean(torch.abs(self.line[i]))
        return loss

    def TV_loss(self, reg):
        loss = 0
        for i in range(self.dim):
            loss += reg(self.plane[i]) * 1e-2  # + reg(self.line[i]) * 1e-3
        return loss


class DensityVM(VMModule):
    def __init__(self, n_comp, gridSize, scale=0.1, dim=3):
        super(DensityVM, self).__init__(n_comp, gridSize, scale, dim)

    def compute(self, xyz_sampled):
        feat = self.compute_feature(xyz_sampled)
        return torch.sum(feat, dim=0)

    def compute_with_grad(self, xyz_sampled):
        feat = self.compute_feature_with_grad(xyz_sampled)
        return torch.sum(feat, dim=0)


class AppVM(VMModule):
    def __init__(self, n_comp, gridSize, app_dim, scale=0.1, dim=3):
        super(AppVM, self).__init__(n_comp, gridSize, scale, dim)
        self.app_dim = app_dim
        self.mat = nn.Linear(sum(n_comp), app_dim, bias=False)

    def compute(self, xyz_sampled):
        feat = self.compute_feature(xyz_sampled)
        return self.mat(feat.T)

    def compute_with_grad(self, xyz_sampled):
        feat = self.compute_feature_with_grad(xyz_sampled)
        return self.mat(feat.T)
