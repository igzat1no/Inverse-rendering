import cv2
import numpy as np
import os
from omegaconf import OmegaConf
import random
from PIL import Image
import scipy.signal
import time
import torch
import torchvision.transforms as T
import torch.nn.functional as F

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def register_operation():
    OmegaConf.register_new_resolver('add', lambda a, b: a + b)
    OmegaConf.register_new_resolver('sub', lambda a, b: a - b)
    OmegaConf.register_new_resolver('mul', lambda a, b: a * b)
    OmegaConf.register_new_resolver('div', lambda a, b: a / b)
    OmegaConf.register_new_resolver('idiv', lambda a, b: a // b)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver('calc_exp_lr_decay_rate', lambda factor, n: factor**(1./n))


def set_device(gpu):
    if gpu == -1:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda:%d' % gpu)
    else:
        device = torch.device('cpu')
        print('Warning: CPU is used because GPU is not available.')
    return device


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()

def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


import torch.nn as nn
class TVLoss(nn.Module):
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



import plyfile
import skimage.measure
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def backup(logdir, args):
    print('Copying source files to logdir...')
    os.makedirs(f'{logdir}/source', exist_ok=True)
    cmd = f'rsync -r --include puop/data --include puop/modeling/models --include *'
    exclusions = ['__pycache__', 'data', 'logs', '*.egg-info', '.vscode', '*.so',
                    '*.a', '.ipynb_checkpoints', 'build', 'bin', '*.ply', 'eigen', 'pybind11',
                    '*.npy', '*.pth', '.git', 'debug', 'tmp', 'wandb', 'LICENSE', 'README.md',
                    '.gitignore', 'requirements.txt', 'LICENSE', 'vis']
    exclusion = ' '.join(['--exclude ' + a for a in exclusions])
    cmd = cmd + ' ' + exclusion
    cmd = cmd + f' . {logdir}/source/'
    cmd = cmd + ' --info=progress2 --no-i-r'
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError('copy files failed!!!!')
    print('Backuping source done!')


scheduler_dict = {
    'ConstantLR': torch.optim.lr_scheduler.ConstantLR,
    'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
    'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
    'LambdaLR': torch.optim.lr_scheduler.LambdaLR,
    'LinearLR': torch.optim.lr_scheduler.LinearLR,
    'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
    'SequentialLR': torch.optim.lr_scheduler.SequentialLR,
    'StepLR': torch.optim.lr_scheduler.StepLR,
}
def build_scheduler(optim, conf):
    scheduler = None
    if conf.name == 'None':
        return scheduler
    elif conf.name == 'SequentialLR':
        schedulers = []
        for sub_conf in conf.schedulers:
            nw_sch = build_scheduler(optim, sub_conf)
            schedulers.append(nw_sch)
        scheduler = scheduler_dict[conf.name](optim, schedulers, **conf.params)
    else:
        scheduler = scheduler_dict[conf.name](optim, **conf.params)
    return scheduler


def seed_everything(seed: int = 42, workers: bool = False) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
    """
    if not isinstance(seed, int):
        seed = int(seed)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    print(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


# cyclic iterator
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def sample_ray(rays_o, rays_d, render_step_size, near_far, aabb, is_train=True, N_samples=-1):
    near, far = near_far
    vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
    rate_a = (aabb[1] - rays_o) / vec
    rate_b = (aabb[0] - rays_o) / vec
    t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

    rng = torch.arange(N_samples)[None].float()
    if is_train:
        rng = rng.repeat(rays_d.shape[-2],1)
        rng += torch.rand_like(rng[:,[0]])
    step = render_step_size * rng.to(rays_o.device)
    interpx = (t_min[...,None] + step)
    rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
    return rays_pts


@torch.no_grad()
def filtering_rays(model, all_rays, all_rgbs, device, chunk=10240 * 5, bbox_only=False):
    print('========> filtering rays ...')
    tt = time.time()
    mask_filtered = []
    N = torch.tensor(all_rays.shape[:-1]).prod()
    idx_chunks = torch.split(torch.arange(int(N)), chunk)

    for idx_chunk in idx_chunks:
        rays_chunk = all_rays[idx_chunk].to(device)
        rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]

        if bbox_only:
            # only consider whether the ray intersects with the bounding box
            vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)  # avoid div 0
            rate_a = (model.aabb[1] - rays_o) / vec
            rate_b = (model.aabb[0] - rays_o) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1)  #.clamp(min=near, max=far)
            t_max = torch.maximum(rate_a, rate_b).amin(-1)  #.clamp(min=near, max=far)
            mask_inbbox = t_max > t_min
        else:
            xyz_sampled = sample_ray(
                rays_o,
                rays_d,
                model.render_step_size,
                model.near_far,
                model.aabb,
                is_train=False,
                N_samples=256,
            )
            mask_inbbox = (model.alphaMask.sample_alpha(xyz_sampled).view(
                xyz_sampled.shape[:-1]) > 0).any(-1)

        mask_filtered.append(mask_inbbox.cpu())

    mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])
    print(mask_filtered.shape)
    print(
        f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}'
    )
    return all_rays[mask_filtered], all_rgbs[mask_filtered], mask_filtered

