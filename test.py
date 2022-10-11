import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio

# configuration
from config_test import get_config

#dataset
from load_dataset import LLFFDataset

#load our model
from nerf import NeRF,Embedding
from model_utils import load_ckpt

#rendering
from rendering import render_rays


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk):
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,          #coarse
                        use_disp,           #disparity sampling
                        0,                  #perturb
                        0,                  #noise
                        N_importance,
                        chunk)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_config()
    w, h = args.img_wh

    kwargs = {'split':'test',
              'root_dir': args.root_dir,            # dataset for original
              'img_wh': tuple(args.img_wh),         # output dim
              'spheric_poses':args.spheric_poses,    # 360
              'test_frames': args.test_frames       # default 120
              }

    #load test dataset, only gives c2w and rays !
    dataset = LLFFDataset(**kwargs)

    # should be compatible with training!
    embedding_xyz = Embedding(args.N_emb_xyz)
    embedding_dir = Embedding(args.N_emb_dir)
    embeddings = {'xyz': embedding_xyz,
                  'dir': embedding_dir}
    
    models = {}
    nerf_coarse = NeRF(in_channels_xyz=6*args.N_emb_xyz+3,in_channels_dir=6*args.N_emb_dir+3)
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    nerf_coarse.cuda().eval()  # no grad
    models['coarse'] = nerf_coarse

    if args.N_importance > 0:
        nerf_fine = NeRF(in_channels_xyz=6*args.N_emb_xyz+3,in_channels_dir=6*args.N_emb_dir+3)
        load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
        nerf_fine.cuda().eval()
        models['fine'] = nerf_fine


    imgs =  []
    depth_maps = []
    dir_name = f'results/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, 
                                    embeddings, 
                                    rays,
                                    args.N_samples, 
                                    args.N_importance, 
                                    args.use_disp,
                                    args.chunk)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        #saving images
        img_pred = np.clip(results[f'rgb_{typ}'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]

        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        #saving temp depths
        depth_maps += [results[f'depth_{typ}'].view(h, w).cpu().numpy()]

    #saving depths images
    min_depth = np.min(depth_maps)
    max_depth = np.max(depth_maps)
    depth_imgs = (depth_maps - np.min(depth_maps)) / (max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
    depth_imgs_ = [cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET) for img in depth_imgs]

    #saving videos
    if not args.no_video:
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.mp4'), imgs, fps=30)
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}_depth.mp4'), depth_imgs_, fps=30)
