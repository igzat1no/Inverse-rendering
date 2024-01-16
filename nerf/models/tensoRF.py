from models.tensorBase import *
from models.renderer import *
from models.decompose_field import DensityLine, AppLine, DensityVM, AppVM

class TensorVM(TensorBase):
    def __init__(self, args, device, aabb, reso_cur):
        super(TensorVM, self).__init__(args, device, aabb, grid_size=reso_cur)

    def init_svd_volume(self, args):
        self.density = DensityVM(args.density.n_comp, self.grid_size)
        self.app = AppVM(args.app.n_comp, self.grid_size, args.app.feature_dim)

    def get_optparam_groups(self, conf, lr_scale=1.0):
        grad_vars = [{'params': self.density.line, 'lr': conf.lr_init * lr_scale},
                     {'params': self.density.plane, 'lr': conf.lr_init * lr_scale},
                     {'params': self.app.line, 'lr': conf.lr_init * lr_scale},
                     {'params': self.app.plane, 'lr': conf.lr_init * lr_scale},
                     {'params': self.app.mat.parameters(), 'lr': conf.lr_basis * lr_scale}]

        if isinstance(self.renderModule, nn.Module):
            grad_vars += [{
                'params': self.renderModule.parameters(),
                'lr': conf.lr_basis * lr_scale,
            }]
        return grad_vars

    def vector_comp_diffs(self):
        return self.density.vectorDiff() + self.app.vectorDiff()


class TensorCP(TensorBase):
    def __init__(self, args, device, aabb, reso_cur):
        super(TensorCP, self).__init__(args, device, aabb, grid_size=reso_cur)

    def init_svd_volume(self, args):
        self.density = DensityLine(args.density.n_comp, self.grid_size)
        self.app = AppLine(args.app.n_comp, self.grid_size, args.app.feature_dim)

    def get_optparam_groups(self, conf, lr_scale=1.0):
        grad_vars = [{
            'params': self.density.param,
            'lr': conf.lr_init * lr_scale,
        }, {
            'params': self.app.param,
            'lr': conf.lr_init * lr_scale,
        }, {
            'params': self.app.mat.parameters(),
            'lr': conf.lr_basis * lr_scale,
        }]
        if isinstance(self.renderModule, nn.Module):
            grad_vars += [{
                'params': self.renderModule.parameters(),
                'lr': conf.lr_basis * lr_scale,
            }]
        return grad_vars
