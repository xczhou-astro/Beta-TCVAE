import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from utils import SelfAttentionModule, ResNetBlock

def encoder(args):
    model = tf.keras.Sequential(name='encoder')
    model.add(layers.InputLayer(input_shape=(args.img_size, args.img_size, args.img_channels)))

    fs = int(args.img_size / 2 ** 5)
    cn = np.log2(args.first_channels).astype(int)
    channels = [int(2 ** x) for x in range(cn, cn + 5)]
    for ii, c in enumerate(channels):
        if args.use_residual_blocks and ii > 0:
            model.add(ResNetBlock(int(c), args, 'down'))
        else:
            model.add(layers.Conv2D(c, 7 if ii == 0 else 4, strides=2, padding='same',
                                    activation=tf.keras.activations.relu,
                                    kernel_initializer=args.initializer,
                                    kernel_regularizer=args.regularizer))
            if args.use_self_attention and ii == len(channels) - int(16 / fs):
                model.add(SelfAttentionModule(c, args))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation=tf.keras.activations.relu,
                            kernel_initializer=args.initializer,
                            kernel_regularizer=args.regularizer))
    
    model.add(layers.Dense(args.z_dim + args.z_dim,
                           kernel_initializer=args.initializer,
                           kernel_regularizer=args.regularizer))
    return model

def decoder(args):
    model = tf.keras.Sequential(name='decoder')
    model.add(layers.InputLayer(input_shape=(args.z_dim,)))
    model.add(layers.Reshape((1, 1, args.z_dim)))

    fs = int(args.img_size / 2 ** 5)
    model.add(layers.Dense(256, activation=tf.keras.activations.relu,
                            kernel_initializer=args.initializer,
                            kernel_regularizer=args.regularizer))
    model.add(layers.Dense(int(fs ** 2 * 32), activation=tf.keras.activations.relu,
                            kernel_initializer=args.initializer,
                            kernel_regularizer=args.regularizer))
    model.add(layers.Reshape((fs, fs, 32)))
    cn = np.log2(args.first_channels).astype(int)
    channels = [int(2 ** x) for x in range(cn + 1, cn + 5)][::-1] + [args.img_channels]
    for ii, c in enumerate(channels):
        if args.use_self_attention and ii == int(16 / fs):
            model.add(SelfAttentionModule(channels[ii - 1], args))
        if args.use_residual_blocks and ii <= (len(channels) - 2):
            model.add(ResNetBlock(int(c) if c > 1 else 1, args, 'up'))
        else:
            model.add(layers.Conv2DTranspose(c, 4 if ii < (len(channels) - 1) else 7,
                                              strides=2, padding='same',
                                              activation=tf.keras.activations.relu if ii < (len(channels) - 1) else None,
                                               kernel_initializer=args.initializer,
                                                 kernel_regularizer=args.regularizer))
    model.add(layers.Conv2DTranspose(args.img_channels, 3, padding='same', strides=1,
                                     kernel_initializer=args.initializer,
                                     kernel_regularizer=args.regularizer))
    return model

class betaTCVAE(tf.keras.Model):
    def __init__(self, args):
        super(betaTCVAE, self).__init__()
        self.args = args
        self.enc = encoder(self.args)
        self.dec = decoder(self.args)
        
        decoded = self.reparameterization_trick(self.enc.output)
        z = self.reparameterization_trick(self.enc.output, True)
        
        
        self.model = tf.keras.Model(inputs=self.enc.input, 
                                    outputs=[decoded, z, self.encode(self.enc.output)])
        self.opt_vae = tf.keras.optimizers.Adam(learning_rate=self.args.lr,
                                                beta_1=self.args.beta1,
                                                beta_2=self.args.beta2,
                                                epsilon=self.args.epsilon)
        
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.args.z_dim))
        return self.decode(eps, apply_sigmoid=True)
    
    def reparameterization_trick(self, x, return_z=False):
        mean, log_squared_scale = self.encode(x)
        z = self.reparameterize(mean, log_squared_scale)
        if return_z:
            return z
        return self.decode(z)
    
    def encode(self, x):
        mean, log_squared_scale = tf.split(x, num_or_size_splits=[self.args.z_dim, self.args.z_dim], axis=1)
        return mean, log_squared_scale
    
    def reparameterize(self, mean, log_squared_scale):
        if self.args.prior.lower() == 'normal':
            return tf.math.add(
                mean,
                tf.math.exp(log_squared_scale / 2) * tf.random.normal(tf.shape(mean), 0, 1),
                )
        if self.args.prior.lower() == 'laplace':
            lp_samples = self.lp_samples(tf.shape(mean))
            return tf.math.add(
                mean,
                tf.math.exp(log_squared_scale / 2) * lp_samples,
                )
        
    @staticmethod
    def lp_one_sided(s, u, b):
        return -tf.math.log(tf.random.uniform(s, u, b))

    def lp_samples(self, s, u=0, b=1):
        one_side = self.lp_one_sided(s, u, b)
        the_other_side = self.lp_one_sided(s, u, b)
        return one_side - the_other_side
    
    def decode(self, z, apply_sigmoid=False):
        z = tf.cast(z, tf.float32)
        logits = self.dec(z)
        if apply_sigmoid:
            probs = tf.math.sigmoid(logits)
            return probs
        return logits