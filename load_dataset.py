import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

# generate rays, poses, and use colmap 
from pose_utils import center_poses, create_spiral_poses, create_spheric_poses
from ray_utils import get_ray_directions, get_rays, get_ndc_rays
from colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary

class LLFFDataset(Dataset):
    def __init__(self, root_dir, split, img_wh=(504, 378), spheric_poses=False,test_frames=120):
        """
        root_dir: processed LLFF data dir
        split: ['train','val','test']
        img_wh: training resolution default (504,378) (half the images_4) (quarter half of the original images)
        spheric_poses: 360 or face forward
        val_num: number of val images
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh   # training resolution
        self.spheric_poses = spheric_poses
        self.val_num = 1 # val image num
        self.define_transforms()    #  self.transform = transforms.ToTensor()  transfer any image to shape c,h,w and normalize
                                    # normalize has must the original type to be unit8
        self.test_frames = test_frames  # the number of testing frames generated. specifically 30 at least for a second video
        self.read_meta()

    def read_meta(self):

        # Step 1: rescale focal length according to training resolution
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))  # this is camera intrinsic
        # camdata[1] is a tuple having:
        #   ["id", "model", "width", "height", "params"]
        #H = camdata[1].height   #hard code here, camdata is a dict has only key named 1
        W = camdata[1].width    #hard code here, camdata is a dict has only key named 1
        self.focal = camdata[1].params[0] * self.img_wh[0]/W    # params: [f cx,cy,k]   here just need the focal
            # our recon image is smaller than original so the focal is smaller too

        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin')) # camera extrinsic
        # imdata is dict with keys numed, and the tuple having:
        # ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
        # id: id in colmap ordered
        # name: original file name
        # q,t: Rt
        perm = np.argsort([imdata[k].name for k in imdata])   # a list of order
        # read successfully reconstructed images and ignore others
        self.image_paths = [os.path.join(self.root_dir, 'images', name) # a list of suce pathes
                            for name in sorted([imdata[k].name for k in imdata])]

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
        poses = c2w_mats[:, :3] # (N_images, 3, 4) cam2world matrices , get rid of the last row
            # to turn the images data from camera specific to world space

        # what is poses ?
        # each image data is specificed by it's pose, so pose(c2w) help all image data focus on
        # the real single scene

        # read bounds
        self.bounds = np.zeros((len(poses), 2)) # (N_images, 2)  # N n f
        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin')) 
        # pts3d is dict of points reconstructed
        # ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
        # image_ids: who saw it
        # 
        pts_world = np.zeros((1, 3, len(pts3d))) # (1, 3, N_points)   location of pts
        visibilities = np.zeros((len(poses), len(pts3d))) # (N_images, N_points)
        for i, k in enumerate(pts3d):
            pts_world[0, :, i] = pts3d[k].xyz
            for j in pts3d[k].image_ids:
                visibilities[j-1, i] = 1
        # calculate each point's depth w.r.t. each camera
        # it's the dot product of "points - camera center" and "camera frontal axis"
        # poses[..., 3:4] last col of all shape (25,3,1)
        # poses[..., 2:3] second last col of all shape (25,3,1)
        depths = ((pts_world-poses[..., 3:4])*poses[..., 2:3]).sum(1) # (N_images, N_points)
        for i in range(len(poses)):
            visibility_i = visibilities[i]
            zs = depths[i][visibility_i==1]
            self.bounds[i] = [np.percentile(zs, 0.1), np.percentile(zs, 99.9)]

        # permute the matrices to increasing order
        poses = poses[perm]
        self.bounds = self.bounds[perm]
        

        # COLMAP poses has rotation in form "right down front", change to "right up back"
        poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)  #neg

        # center the poses
        self.poses, _ = center_poses(poses)  # final result poses

        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
                                                   # center image
        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # COLMAP DATA above:
        # self.focal   f                    for a whole camera
        # self.image_paths
        # self.bounds  (n,f)            
        # self.poses   3*4 Rt




        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (H, W, 3)
            #  focal gives the camera location, similar camera as COLMAP give
            
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            for i, image_path in enumerate(self.image_paths):
                if i == val_idx: # exclude the val image
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                #color
                img = Image.open(image_path).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)  # to our resolution
                img = self.transform(img) # (3, h, w)  to tensor  , and normalized already
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs.append(img)
                
                #rays
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                if self.spheric_poses: #360  
                    near = self.bounds.min()  # use global near,far
                    far = self.bounds.max() # may go less (10*near) for a central focus                
                else:      # need to go NDC for LLFF
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                                     # near plane is always at 1.0
                                     # near and far in NDC are always 0 and 1
                                     # https://github.com/bmild/nerf/issues/34

                all_rays = torch.cat([rays_o, rays_d, \
                                             near*torch.ones_like(rays_o[:, :1]),  # diff from [:,0] and [:,:1] is later one gives a dim
                                             far*torch.ones_like(rays_o[:, :1])],
                                             1)

                self.all_rays.append(all_rays) # (h*w, 8)
                                 
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        
        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.val_idx = val_idx

        else: # test
            if self.spheric_poses: #360
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius,n_poses = self.test_frames)
            else:
                focus_depth = 4   # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 60, axis=0)  #poses: 3*4  camera 2 world trans   25*3*1 -> 3*1  already centered
                self.poses_test = create_spiral_poses(radii, focus_depth,n_poses = self.test_frames) 


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':  
            return len(self.all_rays)
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.poses[self.val_idx])
            else: # test
                c2w = torch.FloatTensor(self.poses_test[idx])
                

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)

            # only gives rays and c2w (only using rays when test)
            sample = {'rays': rays,
                      'c2w': c2w}

            if self.split == 'val':
                idx = self.val_idx
                img = Image.open(self.image_paths[idx]).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img

        return sample
