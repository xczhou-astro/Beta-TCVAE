import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from dataloader import get_data
from model import betaTCVAE
from losses import ae_reconstruction_loss, kl_penalty, tc_penalty
from utils import rate_scheduler, auto_size, split
from evaluation import eval_model, save_checkpoint, plot_summary
from config import get_args
import json

def train(args):

    args.z_dim = int(args.z_dim)
    
    with tf.device('cpu:0'):
        dataset, datasize = get_data(args)

    args.initializer = 'glorot_uniform'
    args.regularizer = tf.keras.regularizers.l2(args.weight_decay)
    
    model = betaTCVAE(args)
    print(model.enc.summary())
    print(model.dec.summary())

    history = {}
    history['loss'] = []
    history['elbo'] = []
    history['ae_loss'] = []
    history['kl_loss'] = []
    history['tc_loss'] = []
    history['tc_estimate'] = []

    args.global_step = 0
    args.steps_per_epoch = datasize // args.batch_size + 1

    for epoch in range(args.epochs):

        loss_ = {}
        loss_['loss'] = []
        loss_['elbo'] = []
        loss_['ae_loss'] = []
        loss_['kl_loss'] = []
        loss_['tc_loss'] = []
        loss_['tc_estimate'] = []
        
        for data in dataset:

            with tf.GradientTape() as vae_tape:

                logits, z, (mean, log_var) = model.model(data)

                args.annealed_beta = rate_scheduler(
                    args.global_step,
                    int(args.steps_per_epoch * args.epochs / 2),
                    args.beta,
                    args.beta
                ) + 1.

                ae_loss = ae_reconstruction_loss(data, logits, True)
                kl_loss = kl_penalty(mean, log_var, args.prior)
                tc_loss, tc = tc_penalty(args, model.reparameterize(mean, log_var),
                                         mean, log_var, args.prior)
                
                ae_loss = tf.reduce_mean(ae_loss)
                kl_loss = tf.reduce_mean(kl_loss)
                tc_loss = tf.reduce_mean(tc_loss)
                tc = tf.reduce_mean(tc)

                elbo = ae_loss + kl_loss
                loss = elbo + tc_loss

            vae_grads = vae_tape.gradient(loss, model.model.trainable_variables)
            model.opt_vae.apply_gradients(zip(vae_grads, model.model.trainable_variables))

            loss_['loss'].append(loss.numpy())
            loss_['elbo'].append(elbo.numpy())
            loss_['ae_loss'].append(ae_loss.numpy())
            loss_['kl_loss'].append(kl_loss.numpy())
            loss_['tc_loss'].append(tc_loss.numpy())
            loss_['tc_estimate'].append(tc.numpy())

        
        loss_e = np.mean(loss_['loss'])
        elbo_e = np.mean(loss_['elbo'])
        ae_loss_e = np.mean(loss_['ae_loss'])
        kl_loss_e = np.mean(loss_['kl_loss'])
        tc_loss_e = np.mean(loss_['tc_loss'])
        tc_estimate_e = np.mean(loss_['tc_estimate'])

        history['loss'].append(float(loss_e))
        history['elbo'].append(float(elbo_e))
        history['ae_loss'].append(float(ae_loss_e))
        history['kl_loss'].append(float(kl_loss_e))
        history['tc_loss'].append(float(tc_loss_e))
        history['tc_estimate'].append(float(tc_estimate_e))

        print(f'Epoch {epoch+1}/{args.epochs}')
        print('loss:', loss_e)
        print('elbo:', elbo_e)
        print('ae_loss:', ae_loss_e)
        print('kl_loss:', kl_loss_e)
        print('tc_loss:', tc_loss_e)
        print('tc_estimate:', tc_estimate_e)
        
        if ((epoch + 1) % 5 == 0) | (epoch == 0):

            output_dir_epoch = args.output_dir + f'/epoch_{epoch+1:03d}'
            os.makedirs(output_dir_epoch, exist_ok=True)

            save_checkpoint(model, epoch, output_dir_epoch)
            
        eval_model(args, model, epoch, output_dir_epoch)

        args.global_step += 1
    
    with open(args.output_dir + '/history.json', 'w') as file:
        json.dump(history, file)
        
    plot_summary(args.output_dir)

    

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    args_dict = vars(args)
    with open(args.output_dir + '/args.json', 'w') as file:
        json.dump(args_dict, file)
    
    gpu_indices = split(args.idx_gpu, int)

    physical_devices = tf.config.list_physical_devices('GPU')

    if physical_devices:
        # Set the desired GPU devices
        visible_devices = [physical_devices[i] for i in gpu_indices]
        tf.config.experimental.set_visible_devices(visible_devices, 'GPU')

        # Allow memory growth for each GPU to avoid allocation errors
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
            
    # strategy = tf.distribute.MirroredStrategy()
    # print(f"Number of devices: {strategy.num_replicas_in_sync}")
    # args.strategy = strategy
    
    train(args)