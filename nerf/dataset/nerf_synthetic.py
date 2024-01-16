import json
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from dataset.utils import get_ray_directions, get_rays
from render.util import scaled_perspective


class NerfSyntheticDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1, is_stack=False, crop=False, debug=False):
        self.data_dir = datadir
        self.split = split
        self.downsample = downsample
        self.img_wh = (800, 800)
        self.is_stack = is_stack
        self.crop = crop
        self.transform = T.ToTensor()
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]])
        self.white_bg = True
        self.near_far = [2.0, 6.0]
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.debug = debug

        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.data_dir, f'transforms_{self.split}.json'), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.w, self.h = w, h
        self.aspect = w / h
        self.n_images = len(self.meta['frames'])
        self.camera_angle_x = self.meta['camera_angle_x']
        self.focal = 0.5 * w / np.tan(0.5 * self.camera_angle_x)
        self.camera_angle_y = np.arctan(np.tan(0.5 * self.camera_angle_x) / self.aspect) * 2.0

        self.directions = get_ray_directions(h, w, [self.focal, self.focal])
        self.directions /= torch.norm(self.directions, dim=-1, keepdim=True)  # (h, w, 3)
        self.intrinsics = torch.tensor([[self.focal, 0, w / 2],
                                    [0, self.focal, h / 2], [0, 0, 1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.all_01_masks = []

        self.all_images = []
        self.all_mvs = []
        self.all_campos = []

        idxs = list(range(0, len(self.meta['frames'])))

        if self.debug:
            idxs = idxs[:2]

        for i in tqdm(idxs, desc=f'Loading {self.split} data'):
            frame = self.meta['frames'][i]
            transform_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            pose = np.array(transform_mat) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.data_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)  # (h, w, 4)
            img = self.transform(img)  # (4, h, w) & normalize to [0,1]

            if self.is_stack:
                mv = torch.linalg.inv(transform_mat)
                campos = transform_mat[:3, 3]
                self.all_mvs.append(mv)
                self.all_campos.append(campos)
                self.all_images.append(img.permute(1, 2, 0))

            img = img.reshape(4, -1).permute(1, 0)  # (h * w, 4) RGBA
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
            # '01_masks': self.all_01_masks[idx],
            'light_idx': self.all_light_idx[idx],
        }
        if self.is_stack:
            sample.update({
                'img': self.all_images[idx],
                'mv': self.all_mvs[idx],
                'campos': self.all_campos[idx],
            })
        return sample

    def get_len(self):
        return self.__len__()

    def get_crop(self, nw_data, cx, cy):
        new_w, new_h = self.w//self.downsample, self.h//self.downsample
        ret = {}
        if nw_data['img'].dim() == 4:
            nw_data['img'] = nw_data['img'].squeeze(0)
        ret['img'] = nw_data['img'][cx - new_w//2:cx + new_w//2, cy - new_h//2:cy + new_h//2, :]
        ret['img'] = ret['img'].unsqueeze(0)
        ret['mv'] = nw_data['mv']
        proj = scaled_perspective(self.camera_angle_y, self.aspect, 0.1, 1000,
                                  self.downsample, cy, cx, new_w, new_h, crop=True)
        ret['mvp'] = proj @ ret['mv']
        ret['campos'] = nw_data['campos']
        return ret


if __name__ == '__main__':
    # print working path
    print(os.getcwd())
    nw = NerfSyntheticDataset('data/nerf_synthetic/lego', split='val', downsample=1, is_stack=False)
    dataloader = DataLoader(nw, batch_size=1, shuffle=True, num_workers=0)
    lzt = iter(dataloader)
    for i in range(1000):
        nw = next(lzt)
        print(nw)