import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from model import decoder
from config import get_args
import copy
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str)
parser.add_argument('--z_dim', type=int)
parser.add_argument('--first_channels', type=int, default=64)
parser.add_argument('--savefilename', type=str)

args = parser.parse_args()

output_dir = args.output_dir
z_dim = args.z_dim
first_channels = args.first_channels
savefilename = args.savefilename

def run_testing(output_dir, z_dim=10, first_channels=64, savefilename=None):
    
    class Args:
        def __init__(self):
            self.z_dim = z_dim
            self.img_size = 64
            self.img_channels = 3
            self.initializer = 'glorot_uniform'
            self.regularizer = tf.keras.regularizers.l2(.01)
            self.use_self_attention = False
            self.use_residual_blocks = True
            self.first_channels = first_channels
            
    args = Args()
            
    dec = decoder(args)
    
    dec.load_weights(os.path.join(output_dir, 'decoder'))
    
    latents = np.zeros([6, z_dim])
    fig, ax = plt.subplots(z_dim, 6, figsize=(5 * 6, 6 * z_dim))
    for i in range(z_dim):
        lat = copy.deepcopy(latents)
        lat[:, i] = np.linspace(-2, 2, 6)
        
        decoded = dec(lat)
        decoded = tf.math.sigmoid(decoded).numpy()
        
        for j in range(6):
            ax[i, j].imshow(decoded[j])
            ax[i, j].axis('off')
    
    if savefilename is not None:
        plt.savefig('figures/' + savefilename)
        
    
run_testing(output_dir, z_dim, first_channels, savefilename)