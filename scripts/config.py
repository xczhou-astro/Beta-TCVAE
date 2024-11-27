import os
import argparse
from utils import str2bool, check_folder

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default='betaTCVAE', help='Name of the model')
    parser.add_argument('--img_size', type=int, default=64, help='Size of the input image')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of channels in the input image')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--z_dim', type=int, default=10, help='Dimension of the latent space')
    parser.add_argument('--prior', type=str, default='normal',
                        help='Prior', choices=['normal', 'laplace'])
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--beta', type=float, default=5,
                        help='beta for the TC loss')
    parser.add_argument('--epsilon', type=float, default=1e-8,)
    parser.add_argument('--weight_decay', type=float, default=.01,
                        help='l2 weight decay')
    parser.add_argument('--use_self_attention', type=str2bool, default=False,)
    parser.add_argument('--use_residual_blocks', type=str2bool, default=True)
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--num_img', type=int, default=6, help='Number of interp images')
    parser.add_argument('--idx_gpu', type=str, default='1', help='Index of GPU to use')
    parser.add_argument('--first_channels', type=int, default=64, help='Number of channels in the first layer of the encoder')
    parser.add_argument('--augmentation', type=str2bool, default=False, help='Whether to use data augmentation')
    parser.add_argument('--dataset', choices=['BGS', 'ZOO'], required=True)

    args = parser.parse_args()

    return args
