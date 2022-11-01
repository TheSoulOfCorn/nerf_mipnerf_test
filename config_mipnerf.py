import argparse

def get_config():
    parser = argparse.ArgumentParser()

#basic
    parser.add_argument("--exp_name", help="name, will be in ckpts and logs", type=str, default='temp_test')
    parser.add_argument("--data_path", help="to the path of COLMAPed LLFF data", type=str, default='/home/mohan/coled_data/flower')

#train and val
    parser.add_argument("--train_batch_size", help="train_batch_size", type=int, default=2048)
    parser.add_argument("--val_chunk_size", help="val_chunk_size", type=int, default=8192) # The amount of input rays in a forward propagation

#nerf
    parser.add_argument("--nerf_resample_padding", help="# Dirichlet/alpha padding on the histogram.", type=float, default=0.01)
    parser.add_argument('--nerf_disparity', default=False, action="store_true",help='use disparity depth sampling')
    parser.add_argument('--nerf_min_deg_point', help="Min degree of positional encoding for 3D points.",type=int,default=0)
    parser.add_argument('--nerf_max_deg_point', help="Max degree of positional encoding for 3D points.",type=int,default=16)
    parser.add_argument('--nerf_deg_view', help=" Degree of positional encoding for viewdirs.",type=int,default=4)
    parser.add_argument('--nerf_density_noise', help=" Standard deviation of noise added to raw density.",type=int,default=0)
    parser.add_argument('--nerf_density_bias', help="The shift added to raw densities pre-activation.",type=int,default=-1)
    parser.add_argument('--nerf_rgb_padding', help="Padding added to the RGB outputs.",type=float,default=0.001)     

#optimizer
    parser.add_argument('--lr', help="The initial learning rate.",type=float,default=5e-4)
    parser.add_argument('--lr_final', help="The final learning rate.",type=float,default=5e-6)
    parser.add_argument('--optimizer_lr_delay_steps', help="The number of warmup learning steps.",type=int,default=2500)
    parser.add_argument('--optimizer_lr_delay_mult', help="How much sever the warmup should be.",type=float,default=0.01)

    parser.add_argument("--max_steps", help="max train steps", type=int, default=200000)

#loss
    parser.add_argument("--loss_coarse_loss_mult", help="loss_coarse_loss_mult", type=float, default=0.1)

    return parser.parse_args()
