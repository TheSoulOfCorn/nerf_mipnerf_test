import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", help="Path to ckpt.",default='/home/mohan/mipnerf/ckpts/library_test/epoch=085.ckpt')
    parser.add_argument("--data_dir", help="Original data directory.", type=str, default='/home/mohan/coled_data/library')
    parser.add_argument("--out_dir", help="Output directory.", type=str, default='/home/mohan/mipnerf/result_mip')
    parser.add_argument("--exp_name", help="example name.", type=str, default='temp_test')
    parser.add_argument("--chunk_size", help="Chunck size for render.", type=int, default=4096)
    parser.add_argument('--img_wh', help='the output size you want', type=list, default=[504,378])

    return parser.parse_args()
