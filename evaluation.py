import os
import numpy as np
import tensorflow as tf
from itertools import combinations, product
from scipy.stats import norm, laplace
from matplotlib.pyplot import close
from sklearn import metrics
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
from dataloader import get_data
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from skimage.metrics import structural_similarity as ssim
import json

def image_grid(img, grid_size=25, ttl=None):
    sns.set_style('white')
    if not isinstance(img, (np.ndarray, np.generic)):
        img = np.array(img)
    
    f = plt.figure(figsize=(10, 10))
    for i in range(grid_size):
        ax = plt.subplot(int(np.sqrt(grid_size)), int(np.sqrt(grid_size)), i+1)
        ax.imshow(img[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.axis('off')
    if ttl is not None:
        f.suptitle(ttl, fontsize=20)
    plt.box(False)
    f.tight_layout(pad=0., h_pad=0., w_pad=0.)
    return f

def save_checkpoint(model, epoch, output_dir_epoch):
    model.enc.save_weights(os.path.join(output_dir_epoch, 'encoder'))
    model.dec.save_weights(os.path.join(output_dir_epoch, 'decoder'))
    # try:
    #     model.enc.save(os.path.join(output_dir_epoch, 'encoder.keras'))
    #     model.dec.save(os.path.join(output_dir_epoch, 'decoder.keras'))
    # except:
    #     pass
    
def plot_summary(output_dir):
    
    with open(output_dir + '/history.json', 'r') as file:
        his = json.load(file)
    
    keys = his.keys()
    for key in keys:
        plt.figure()
        plt.plot(his[key])
        plt.savefig(output_dir + f'/{key}.png')
    

def eval_model(args, model, epoch, output_dir_epoch):

    if ((epoch + 1) % 5 == 0) | (epoch == 0):
        
        test_dataset, test_datasize = get_data(args, 'test')

        idx = np.random.choice(test_datasize, 3, replace=False)

        x_true = []
        x_pred = []

        for ii, x_test in enumerate(test_dataset):
            logits, _, (z_mean, log_var) = model.model(x_test)
            xhat = tf.math.sigmoid(logits)

            x_true.append(x_test.numpy())
            x_pred.append(xhat.numpy())

        x_true = np.concatenate(x_true, axis=0)
        x_pred = np.concatenate(x_pred, axis=0)
        
        
        ssim_metrics = []
        for xt, xp in zip(x_true, x_pred):
            ssim_metric = ssim(xt, xp, data_range=1., 
                            channel_axis=2)
            ssim_metrics.append(ssim_metric)
            
        ssim_mean = np.mean(ssim_metrics)
        
        print(f'mean SSIM: {np.around(ssim_mean, 3)}')

        x_true = x_true[idx]
        x_pred = x_pred[idx]
        

        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        
        for i in range(3):
            ax[0, i].imshow(x_true[i])
            ax[0, i].axis('off')
            ax[1, i].imshow(x_pred[i])
            ax[1, i].axis('off')
        
        plt.suptitle(f'SSIM = {np.around(ssim_mean, 3)}', fontsize=30)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_epoch, 'examples.png'))
        plt.close()
        
        
        latents = np.zeros([args.num_img, args.z_dim])
        
        for i in range(args.z_dim):
            lat = copy.deepcopy(latents)
            lat[:, i] = np.linspace(-2, 2, args.num_img)
            inp = lat[..., np.newaxis, np.newaxis]
            decoded = model.decode(inp, apply_sigmoid=True).numpy()
            fig, ax = plt.subplots(1, args.num_img, figsize=(5 * args.num_img, 5))
            for j in range(args.num_img):
                ax[j].imshow(decoded[j])
                ax[j].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir_epoch, 'interp_{}.png'.format(i)))
            plt.close()