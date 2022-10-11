import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # BASIC---------------------------------------------------------------------------------
    parser.add_argument('--root_dir', type=str,default='./data',
                        help='root directory of dataset')
    # mostly needed is the COLMAP pos info, dont need to original color anymore
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')
    # to the ckpt direction #may want to specific the model name (only the best one)
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, just for output name setting')
    # final result will be in ./result/[scene_name]
    parser.add_argument('--img_wh', nargs="+", type=int, default=[504,378],
                        help='resolution (img_w, img_h) of the image')
    # any reso you want, keep same ratio if dont want distortion
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')
    # if 360

    # NERF DETAIL---------------------------------------------------------------------------
    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of frequencies in xyz positional encoding')
    # should be the same with training!
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of frequencies in dir positional encoding')
    # should be the same with training!
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')
    
    # SAVE OPTION---------------------------------------------------------------------------
    parser.add_argument('--test_frames',type = int, default=120,
                        help='frame number of testing result')
    parser.add_argument('--no_video', default=False, action="store_true",
                        help='will not save video for depth and color image')

    return parser.parse_args()
