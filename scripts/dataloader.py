import os
import numpy as np
import tensorflow as tf
from PIL import Image

def tf_transform(img, n):
    if n == 0:
        transform = tf.image.rot90(img, k=0)
    elif n == 1:
        transform = tf.image.rot90(img, k=1)
    elif n == 2:
        transform = tf.image.rot90(img, k=2)
    elif n == 3:
        transform = tf.image.flip_left_right(img)
    elif n == 4:
        transform = tf.image.flip_up_down(img)
    elif n == 5:
        transform = tf.image.rot90(tf.image.flip_left_right(img), k=1)
    elif n == 6:
        transform = tf.image.rot90(tf.image.flip_up_down(img), k=1)
    else:
        transform = img
    
    return transform

def load_img_aug(img_path, transform_n):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if img.shape[0] != 64:
        img = tf.image.resize(img, [64, 64])
    img = tf_transform(img, transform_n)
    
    return img

def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if img.shape[0] != 64:
        img = tf.image.resize(img, [64, 64])
    return img

def get_data(args, split='train'):

    if args.dataset == 'BGS':
        data_dir = '../FactorVAE/data/BGS_ALL_jpg/BGS'
    elif args.dataset == 'ZOO':
        data_dir = '../dataset/images_training_rev1'
    else:
        pass
    
    
    img_ls = os.listdir(data_dir)
    datasize = len(img_ls)
    test_num = int(0.1 * datasize)
    train_num = datasize - test_num
    if split == 'train':
        img_ls = img_ls[:train_num]
    elif split == 'test':
        img_ls = img_ls[train_num:]
    else:
        raise ValueError('Invalid split')
    
    datasize = len(img_ls)

    img_paths = [os.path.join(data_dir, img_name) for img_name in img_ls]
    
    if split == 'train' and args.augmentation:
        img_paths = np.array(img_paths)
        idx = np.arange(datasize)
        img_paths = np.repeat(img_paths, 8)
        transform_n = np.tile(np.arange(8), datasize)
        
        datasize = datasize * 8
        shuffle_idx = np.random.choice(datasize,
                                       datasize,
                                       replace=False)
        
        img_paths = img_paths[shuffle_idx]
        transform_n = transform_n[shuffle_idx]
        
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, transform_n))
        dataset = dataset.map(lambda img_path, transform_n: load_img_aug(img_path, transform_n), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            
    else:
    
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
        dataset = dataset.map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, datasize