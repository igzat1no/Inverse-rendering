from models.basemodel_dmtet import Basemodel_dmtet
from models.tensorBase import *
from models.renderer import *
from models.myutils import *
from models.tensoIR.relight_utils import *
from utils import TVLoss

from geometry import dmtet
from render import mesh, regularizer, render


class TensoIR_DMTet(Basemodel_dmtet):
    def cal_sdf(self, v):
        sdf = self.sdf_net.compute_alpha(v, self.sdf_net.render_step_size) * \
            self.init_scale - self.get_density_threshold()
        return sdf

    def cal_loss(self, buffers, target, loss_fn, iteration, mat=None):
        color_ref = target['img']

        total_loss = 0
        loss_dict = {}

        albedo = buffers['albedo'] # 1 x H x W x 3
        # mse loss between albedo and albedo_jitter
        albedo_smoothness_loss = torch.nn.functional.mse_loss(albedo, buffers['albedo_jitter'])
        roughness_smoothness_loss = torch.nn.functional.mse_loss(buffers['roughness'], buffers['roughness_jitter'])

        if mat is not None:
            mat_net = mat['neural_tex'].net
            l1_reg_loss = mat_net.density_L1() * self.conf.l1_reg_weight
            total_loss += l1_reg_loss
            loss_dict['l1_reg_loss'] = l1_reg_loss

            loss_tv_density = mat_net.TV_loss_density(mat_net.tvreg) * self.conf.tv_weight_density
            total_loss += loss_tv_density
            loss_dict['loss_tv_density'] = loss_tv_density

            loss_tv_app = mat_net.TV_loss_app(mat_net.tvreg) * self.conf.tv_weight_app
            total_loss += loss_tv_app
            loss_dict['loss_tv_app'] = loss_tv_app

            roughness_smoothness_loss = self.conf.BRDF_loss_enhance_ratio * self.conf.roughness_smoothness_loss_weight * roughness_smoothness_loss
            total_loss += roughness_smoothness_loss
            loss_dict['roughness_smoothness_loss'] = roughness_smoothness_loss

            albedo_smoothness_loss = self.conf.BRDF_loss_enhance_ratio * self.conf.albedo_smoothness_loss_weight * albedo_smoothness_loss
            total_loss += albedo_smoothness_loss
            loss_dict['albedo_smoothness_loss'] = albedo_smoothness_loss

            normals_diff_loss = self.conf.normals_loss_enhance_ratio * self.conf.normals_diff_weight * buffers['normals_diff'].mean()
            total_loss += normals_diff_loss
            loss_dict['normals_diff_loss'] = normals_diff_loss

            normals_orientation_loss = self.conf.normals_loss_enhance_ratio * self.conf.normals_orientation_weight * buffers['normals_orientation_loss'].mean()
            total_loss += normals_orientation_loss
            loss_dict['normals_orientation_loss'] = normals_orientation_loss

        # import imageio
        # vis_albedo = albedo.cpu().detach().numpy()
        # vis_albedo = (vis_albedo * 255).astype(np.uint8)
        # imageio.imwrite('albedo12.png', vis_albedo)
        # vis_img = (buffers['shaded'][..., 0:3] * color_ref[..., 3:])[0].cpu().detach().numpy()
        # vis_img = np.clip(vis_img, 0, 1)
        # vis_img = (vis_img * 255).astype(np.uint8)
        # imageio.imwrite('img.png', vis_img)
        # for i in range(10):
        #     mask = albedo_masks[i][0].cpu().detach().numpy()
        #     mask = (mask * 255).astype(np.uint8)
        #     mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        #     imageio.imwrite('mask_{}.png'.format(i), mask)

        if color_ref.shape[-1] == 4:
            img_loss = loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
            img_loss = img_loss + torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        else:
            img_loss = loss_fn(buffers['shaded'][..., 0:3], color_ref[..., 0:3])
        total_loss += img_loss
        loss_dict['img_loss'] = img_loss

        if abs(self.conf.normal_smooth_weight) > 1e-4:
            reg_loss = regularizer.normal_consistency(self.opt_mesh.v_pos, self.opt_mesh.t_pos_idx) * self.conf.normal_smooth_weight
        else:
            reg_loss = torch.Tensor([0.0]).cuda()


        loss_dict['reg_loss'] = reg_loss
        loss_dict['total_loss'] = total_loss

        return loss_dict