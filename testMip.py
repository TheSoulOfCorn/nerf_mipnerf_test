import os
import argparse
import collections
import torchvision
import torchvision.transforms as T
import imageio
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from config_mipnerf_test import get_config
from mip_nerf import MipNerf
from model_utils import load_ckpt
from pose_utils import center_poses,create_spiral_poses
from ray_utils import get_rays_mip,get_ray_directions,get_ndc_rays
from colmap_utils import read_cameras_binary, read_images_binary

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
Rays_keys = Rays._fields

def save_image_tensor(image, height, width, save_path, nhwc = True):
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.dim() == 3:
        image = image[None, ...]
        if nhwc:  # nhwc -> nchw
            image = image.permute(0, 3, 1, 2)
        torchvision.utils.save_image(image, save_path)
    elif image.dim() == 4:
        if nhwc:  # nhwc -> nchw
            image = image.permute(0, 3, 1, 2)
        torchvision.utils.save_image(image, save_path)
    elif image.dim() == 2:  # flatten
        assert image.shape[0] == height * width
        image = image.reshape(1, height, width, image.shape[-1])
        if nhwc:  # nhwc -> nchw
            image = image.permute(0, 3, 1, 2)
        torchvision.utils.save_image(image, save_path)
    else:
        raise NotImplementedError

def visualize_depth(depth):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = T.ToTensor()(x)  # (3, H, W)
    return x_

def save_images(rgb, dist, path, idx):
    B, H, W, C = rgb.shape
    color_dist = visualize_depth(dist)
    save_image_tensor(rgb, H, W, os.path.join(path, str('{:05d}'.format(idx)) + '_rgb' + '.png'))
    save_image_tensor(color_dist, H, W, os.path.join(path, str('{:05d}'.format(idx)) + '_dist' + '.png'), False)

def rearrange_render_image(rays, chunk_size=4096):
    # change Rays to list: [origins, directions, viewdirs, radii, lossmult, near, far]
    single_image_rays = [getattr(rays, key) for key in Rays_keys]
    val_mask = single_image_rays[-3]

    # flatten each Rays attribute and put on device
    single_image_rays = [rays_attr.reshape(-1, rays_attr.shape[-1]) for rays_attr in single_image_rays]
    # get the amount of full rays of an image
    length = single_image_rays[0].shape[0]
    # divide each Rays attr into N groups according to chunk_size,
    # the length of the last group <= chunk_size
    single_image_rays = [[rays_attr[i:i + chunk_size] for i in range(0, length, chunk_size)] for
                         rays_attr in single_image_rays]
    # get N, the N for each Rays attr is the same
    length = len(single_image_rays[0])
    # generate N Rays instances
    single_image_rays = [Rays(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]
    return single_image_rays, val_mask

class RenderGen(Dataset):
    def __init__(self, img_wh, data_dir, N_poses):
        super(RenderGen, self).__init__()
        self.img_wh = img_wh
        self.data_dir = data_dir
        self.N_poses = N_poses
        self._generate_rays()

    def _generate_rays(self):
        """Generating rays for all images."""
        camdata = read_cameras_binary(os.path.join(self.data_dir, 'sparse/0/cameras.bin'))
        W = camdata[1].width
        self.focal = camdata[1].params[0] * self.img_wh[0]/W

        imdata = read_images_binary(os.path.join(self.data_dir, 'sparse/0/images.bin'))
        perm = np.argsort([imdata[k].name for k in imdata])
        w2c_mats = []   # world to camera mats
        fill_line = np.array([0, 0, 0, 1.]).reshape(1, 4)  # last row
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            m = np.concatenate([np.concatenate([R, t], 1), fill_line], 0)
            w2c_mats.append(m)              
        w2c_mats = np.stack(w2c_mats, 0)    # (N_images,4,4) extrinsic
        c2w_mats = np.linalg.inv(w2c_mats)  # (N_images,4,4)
        poses = c2w_mats[:, :3]
        poses = poses[perm]
        poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        self.poses, _ = center_poses(poses)
        self.poses[..., 3] /= 0.75

        # get test poses
        r = np.percentile(np.abs(self.poses[..., 3]), 10, axis=0)
        self.poses_test = create_spiral_poses(r, 20, n_poses = self.N_poses) 

        # dir
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal)


        origins=[]
        dirs=[]
        radii=[]
        self.lossmult=[]
        self.near=[]
        self.far=[]

        for i in range(len(self.poses_test)):

            c2w = torch.FloatTensor(self.poses_test[i])

            rays_o, rays_d = get_rays_mip(self.directions, c2w) #(H,W,3)
            near, far = 0, 1
            #go NDC
            rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                            self.focal, 1.0, rays_o, rays_d)
            origins.append(np.array(rays_o))
            dirs.append(np.array(rays_d))
            self.lossmult.append(np.ones_like(rays_o[...,:1]))
            self.near.append(np.zeros_like(rays_o[...,:1]))
            self.far.append(np.ones_like(rays_o[...,:1]))

        viewdirs = [ v / np.linalg.norm(v, axis=-1, keepdims=True) for v in dirs ]
        dx = [ np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in dirs ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]       


        self.rays = Rays(
            origins=origins,
            directions=dirs,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=self.lossmult,
            near=self.near,
            far=self.far)

        del origins, dirs, viewdirs, radii

    def __len__(self):
        return self.N_poses

    def __getitem__(self, index):
        rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
        return rays

def run_render(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MipNerf(
            resample_padding=0.01,
            disparity=False,
            min_deg_point=0,
            max_deg_point=16,
            deg_view=4,
            density_noise=0,
            density_bias=-1,
            rgb_padding=0.001,
            loading_model=True)
    load_ckpt(model, args.ckpt, model_name='mip_nerf')
    model.cuda().eval()

    # get dir
    out_file_path = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(out_file_path,exist_ok=True)

    # get data
    render_dataset = RenderGen(args.img_wh,args.data_dir,120)
    render_loader = DataLoader(render_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    with torch.no_grad():
        for idx, rays in enumerate(tqdm(render_loader)):
            rays = Rays(*[getattr(rays, name).float().to(device) for name in Rays_keys])
            _, height, width, _ = rays.origins.shape  # N H W C
            single_image_rays, _ = rearrange_render_image(rays, args.chunk_size)
            fine_rgb = []
            distances, accs = [], []
            with torch.no_grad():
                for batch_rays in single_image_rays:
                    _, (f_rgb, distance) = model(batch_rays, False)
                    fine_rgb.append(f_rgb)
                    distances.append(distance)

            fine_rgb = torch.cat(fine_rgb, dim=0)
            distances = torch.cat(distances, dim=0)

            fine_rgb = fine_rgb.reshape(1, height, width, fine_rgb.shape[-1])  # N H W C
            distances = distances.reshape(height, width)  # H W
            save_images(fine_rgb, distances, out_file_path, idx)

if __name__ == '__main__':
    hparams = get_config()  # to hyper parameters to hparams
    run_render(hparams)
