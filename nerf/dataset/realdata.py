import json
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from dataset.utils import get_ray_directions, get_rays
# from utils import get_ray_directions, get_rays

def scaled_perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, downsample=1.0, cx=0, cy=0, h=1, w=1, device=None):
    y = np.tan(fovy / 2)/downsample
    x0 = 2*cx/w - downsample
    y0 = 2*cy/h - downsample
    return torch.tensor([[1/(y*aspect),    0,            x0,              0],
                         [           0, 1/-y,            y0,              0],
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)


class RealDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4.0, is_stack=False):
        self.data_dir = datadir
        self.split = split
        self.downsample = downsample
        self.img_wh = (int(4000/downsample), int(6000/downsample))
        self.is_stack = is_stack
        self.transform = T.ToTensor()
        self.scene_bbox = torch.tensor([[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]])
        self.blender2opencv = np.array([[1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]])
        self.white_bg = True
        self.near_far = [0.2, 4.0]
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.data_dir, f'transforms_{self.split}.json'), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.w, self.h = w, h
        self.aspect = w / h
        self.n_images = len(self.meta['frames'])

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.all_01_masks = []

        self.all_images = []
        self.all_mvs = []
        self.all_mvps = []
        self.all_campos = []

        idxs = list(range(0, len(self.meta['frames'])))

        for i, key in tqdm(enumerate(self.meta['frames'])):
            # if i == 2:
            #     break
            frame = self.meta['frames'][key]
            self.focal = 0.5 * w / np.tan(0.5 * frame['camera_angle_x'])

            self.directions = get_ray_directions(h, w, [self.focal, self.focal])
            self.directions /= torch.norm(self.directions, dim=-1, keepdim=True)  # (h, w, 3)
            self.intrinsics = torch.tensor([[self.focal, 0, w / 2],
                                        [0, self.focal, h / 2], [0, 0, 1]]).float()

            transform_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            pose = np.array(transform_mat) # @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.data_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)  # (h, w, 4)
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)

            img = self.transform(img)  # (4, h, w) & normalize to [0,1]

            if self.is_stack:
                mv = torch.linalg.inv(transform_mat)
                camera_angle_y = np.arctan(np.tan(0.5 * frame['camera_angle_x']) / self.aspect) * 2.0
                proj = scaled_perspective(camera_angle_y, self.aspect, 0.1, 1000,
                                          self.downsample, self.w // 2, self.h // 2, 800, 800)
                campos = transform_mat[:3, 3]
                mvp = proj @ mv
                self.all_mvs.append(mv)
                self.all_mvps.append(mvp)
                self.all_campos.append(campos)
                self.all_images.append(img.permute(1, 2, 0))

            img = img.view(4, -1).permute(1, 0)  # (h * w, 4) RGBA
            self.all_masks += [img[:, -1]]  # (h * w, 1)
            img_mask = ~(img[:, -1] == 0)
            self.all_01_masks += [img_mask.squeeze(0)]
            if self.white_bg:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            else:
                img = img[:, :3] * img[:, -1:]
            self.all_rgbs += [img]  # (h * w, 3)

            # both (h * w, 3), origin and direction for each ray
            rays_o, rays_d = get_rays(self.directions, c2w)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h * w, 6)
        self.poses = torch.stack(self.poses)  # (number of frames, 4, 4)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (number of frames * h * w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (number of frames * h * w, 3)
            self.all_masks = torch.cat(self.all_masks, 0)  # (number of frames * h * w, 1)
            self.all_01_masks = torch.cat(self.all_01_masks, 0)  # (number of frames * h * w, 1)
            self.all_light_idx = torch.zeros((*self.all_rays.shape[:-1], 1), dtype=torch.long)
        else:
            # (number of frames, h * w, 3)
            self.all_rays = torch.stack(self.all_rays, 0).reshape(
                                -1, *self.img_wh[::-1], 6)
            # (number of frames, h, w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
                                -1, *self.img_wh[::-1], 3)
            # (number of frames, h, w, 1)
            self.all_masks = torch.stack(self.all_masks, 0).reshape(
                                -1, *self.img_wh[::-1], 1)
            self.all_01_masks = torch.stack(self.all_01_masks, 0).reshape(
                                -1, *self.img_wh[::-1], 1)
            self.all_light_idx = torch.zeros((*self.all_rays.shape[:-1], 1),dtype=torch.long).reshape(-1,*self.img_wh[::-1])

        # define projection matrix from world coordinate to 2d
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        sample = {
            'rays': self.all_rays[idx],
            'rgbs': self.all_rgbs[idx],
            'masks': self.all_masks[idx],
            '01_masks': self.all_01_masks[idx],
            'light_idx': self.all_light_idx[idx],
        }
        if self.is_stack:
            sample.update({
                'img': self.all_images[idx],
                'mv': self.all_mvs[idx],
                'mvp': self.all_mvps[idx],
                'campos': self.all_campos[idx],
            })
        return sample

    def get_len(self):
        return self.__len__()


if __name__ == '__main__':
    # print working path
    print(os.getcwd())
    nw = RealDataset('data/realdata', split='val', downsample=1.0, is_stack=True)
    dataloader = DataLoader(nw, batch_size=1, shuffle=False, num_workers=0)
    lzt = iter(dataloader)
    for i in range(2):
        nw = next(lzt)
        rays_o, rays_d = nw['rays'][:, :3], nw['rays'][:, 3:]
        print(rays_o[:10], rays_d[:10])
        # visualize 01_masks (h,w,1)
        # save to png
        # import matplotlib.pyplot as plt
        # # extend from (h,w,1) to (h,w,3)
        # nw['01_masks'] = nw['01_masks'] * torch.tensor([1, 1, 1], dtype=torch.float32)
        # plt.imsave('01_masks.png', nw['01_masks'].squeeze(0).numpy())
