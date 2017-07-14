"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from scipy import misc

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader_MultiClass_Loss, prepare_label, decode_labels, inv_preprocess

n_classes = 21

DATA_DIRECTORY = '/home/VOCdevkit'
DATA_LIST_PATH = './dataset/val.txt'
NUM_STEPS = 1449 # Number of images in the validation set.
RESTORE_FROM = './deeplab_resnet.ckpt'

RANDOM_SEED = 1234
EPSILON = 1e-12
def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader_MultiClass_Loss(
            args.data_dir,
            args.data_list,
            None, # No defined input size.
            RANDOM_SEED,
            False, # No random scale.
            False, # No random mirror.
            coord)
        image, l2_catg, binary_catg, hinge_catg = reader.image, reader.l2_catg, reader.binary_catg, reader.hinge_catg
    image_batch = tf.expand_dims(image, dim=0)
    binary_catg_batch = tf.expand_dims(binary_catg, dim=0)

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = net.layers['fc1_voc12']

    # Do the global average pooling
    raw_output_bcgd_rmvd = raw_output[:,:,:,1:]
    g_avg_pool = tf.reduce_mean(tf.reduce_mean(raw_output_bcgd_rmvd, axis=1, keep_dims=True),\
                           axis=2, keep_dims=True) # Avg across the width and height dimension -> [Bx21]
    g_avg_pool_sqzd = tf.squeeze(g_avg_pool, axis=[1, 2])
    pred = tf.nn.softmax(g_avg_pool_sqzd)

    # Get the class activation map
    raw_output_up = tf.image.resize_bilinear(raw_output_bcgd_rmvd, tf.shape(image_batch)[1:3,])
    raw_output_up = raw_output_up - tf.reduce_min(tf.reduce_min(raw_output_up, axis=1, keep_dims=True), axis=2, keep_dims=True) + EPSILON
    raw_output_up = raw_output_up / tf.reduce_max(tf.reduce_max(raw_output_up, axis=1, keep_dims=True), axis=2, keep_dims=True)
    cam_m_1 = tf.argmax(raw_output_up, dimension=3) + 1
    raw_output_catgs_rmvd = raw_output_up * tf.expand_dims(tf.expand_dims(binary_catg_batch, 1), 2)
    cam_m_2 = tf.argmax(raw_output_catgs_rmvd, dimension=3) + 1
    cam = tf.cast(tf.equal(cam_m_1, cam_m_2), tf.int64) * cam_m_1

    cam_batch = tf.expand_dims(cam, dim=3)

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # Iterate over training steps.
    for step in range(args.num_steps):
        preds, images, cams, bin_catg = sess.run([pred, image_batch, cam_batch, binary_catg])
        """
        print(bin_catg)
        print(np.unique(np.unique(cams)))
        """
        img = inv_preprocess(images)
        attMap = decode_labels(cams)
        output_dir = './output_maps_binary_without_norm/'
        img_name = output_dir + str(step) + '.jpg'
        map_name = output_dir + str(step) + '.png'
        misc.imsave(img_name, img[0,:,:,:])
        misc.imsave(map_name, attMap[0,:,:,:])
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
