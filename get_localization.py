"""Script for obtaining localization cues for DCSP.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader_Classfc, inv_preprocess, dense_crf

DATA_DIRECTORY = '/home/VOCdevkit'
DATA_LIST_PATH = './dataset/train.txt'
RESTORE_FROM = './deeplab_resnet.ckpt'
SAVE_DIR = './Attentions'
EPSILON = 1e-12

NUM_CLASSES = 21
def save(saver, sess, logdir, step):
   '''Save weights.
   
   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def get_images_list(data_list):
    """
    data list: /path/to/jpeg /path/to/attentions /path/to/saliency /path/to/classcategories
    """
    f = open(data_list, 'r')
    image_names = []
    for line in f:
        img, _ = line.strip('\n').split(' ')
        image_names.append(img)

    f.close()

    return image_names


def main(data_dir=DATA_DIRECTORY, data_list=DATA_LIST_PATH, restore_from=RESTORE_FROM,\
         save_dir=SAVE_DIR, n_classes=NUM_CLASSES, adapt=False):
    """Create the model and obtain the localization cues."""

    graph = tf.Graph()

    with graph.as_default():

        # Create queue coordinator.
        coord = tf.train.Coordinator()
    
        # Load reader for training.
        with tf.name_scope("create_inputs"):
            reader = ImageReader_Classfc(
                data_dir,
                data_list,
                None,
                1234,
                False,
                False,
                n_classes,
                coord)
            image, catg_with_bcgd, catg_wo_bcgd = reader.image, reader.catg_with_bcgd, reader.catg_wo_bcgd

        image_batch, catg_batch_with_bcgd, catg_batch_wo_bcgd = tf.expand_dims(image, dim=0),\
                tf.expand_dims(catg_with_bcgd, dim=0), tf.expand_dims(catg_wo_bcgd, dim=0) # Add one batch dimension.
        h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
        image_batch075 = tf.image.resize_images\
                (image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
        image_batch05 = tf.image.resize_images\
                (image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.5)), tf.to_int32(tf.multiply(w_orig, 0.5))]))

        # Create network.
        with tf.variable_scope('', reuse=False):
            net = DeepLabResNetModel({'data': image_batch}, is_training=False)
        with tf.variable_scope('', reuse=True):
            net075 = DeepLabResNetModel({'data': image_batch075}, is_training=False)
        with tf.variable_scope('', reuse=True):
            net05 = DeepLabResNetModel({'data': image_batch05}, is_training=False)

        # For a small batch size, it is better to keep 
        # the statistics of the BN layers (running means and variances)
        # frozen, and to not update the values provided by the pre-trained model. 
        # If is_training=True, the statistics will be updated during the training.
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.

        # Which variables to load.
        restore_var = tf.global_variables()
   
        # Predictions.
        raw_output100_init = net.layers['fc1_voc12_d0']
        raw_output075_init = tf.image.resize_images(net075.layers['fc1_voc12_d0'], tf.shape(raw_output100_init)[1:3,])
        raw_output05_init = tf.image.resize_images(net05.layers['fc1_voc12_d0'], tf.shape(raw_output100_init)[1:3,])
        raw_output_init = tf.reduce_max(tf.stack([raw_output100_init, raw_output075_init, raw_output05_init]), axis=0)

        # Predictions.
        raw_output100_adapt = net.layers['fc1_voc12']
        raw_output075_adapt = tf.image.resize_images(net075.layers['fc1_voc12'], tf.shape(raw_output100_adapt)[1:3,])
        raw_output05_adapt = tf.image.resize_images(net05.layers['fc1_voc12'], tf.shape(raw_output100_adapt)[1:3,])
        raw_output_adapt = tf.reduce_max(tf.stack([raw_output100_adapt, raw_output075_adapt, raw_output05_adapt]), axis=0)

        catg_vec_with_bcgd = tf.expand_dims(tf.expand_dims(catg_batch_with_bcgd, dim=1), dim=2)
        catg_vec_wo_bcgd = tf.expand_dims(tf.expand_dims(catg_batch_wo_bcgd, dim=1), dim=2)

        # Calculate the segmentation mask 
        raw_output_up_init = tf.image.resize_bilinear(raw_output_init, tf.shape(image_batch)[1:3,])
        raw_output_up_adapt = tf.image.resize_bilinear(raw_output_adapt, tf.shape(image_batch)[1:3,])

        # Initial attention cues
        raw_output_up_init = raw_output_up_init - tf.reduce_min(tf.reduce_min(raw_output_up_init, axis=1, keep_dims=True),\
                                                      axis=2, keep_dims=True) + EPSILON
        raw_output_up_init = raw_output_up_init / tf.reduce_max(tf.reduce_max(raw_output_up_init, axis=1, keep_dims=True),\
                                                      axis=2, keep_dims=True)
        attention_init = raw_output_up_init * catg_vec_wo_bcgd
        local_cues_init = tf.squeeze(attention_init, axis=0) # Remove the batch dimension

        # Adaptive attention cues
        attention_adapt = raw_output_up_adapt * catg_vec_with_bcgd
        attention_adapt = tf.argmax(attention_adapt, axis=3)
        local_cues_adapt = tf.expand_dims(attention_adapt, dim=3) # Add the channel dimension
        local_cues_adapt = tf.squeeze(local_cues_adapt, axis=0) # Remove the batch dimension

        indices_with_bcgd = tf.cast(tf.squeeze(tf.where(tf.greater(catg_with_bcgd, 0.0)), 1), tf.int32)
        indices_wo_bcgd = tf.cast(tf.squeeze(tf.where(tf.greater(catg_wo_bcgd, 0.0)), 1), tf.int32)

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=graph) as sess:

        # Initialize the model parameters
        tf.global_variables_initializer().run()

        loader = tf.train.Saver(var_list=restore_var)
        if restore_from is not None:
            load(loader, sess, restore_from)
 
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        images_list = get_images_list(data_list)

        for i in range(len(images_list)):
            if adapt:
                attentions, catgs = sess.run([local_cues_adapt, indices_with_bcgd])
                final_attentions = attentions
            else:
                attentions, catgs = sess.run([local_cues_init, indices_wo_bcgd])
                final_attentions = attentions[:,:,catgs]
            base_fname = images_list[i].strip("\n").rsplit('/', 1)[1].replace('jpg', 'npz')
            f_name = save_dir + "/" + base_fname
            np.savez(f_name, actv=final_attentions)

            if(i%1000 == 0):
                print('Processed {}/{}'.format(i, len(images_list)))

        coord.request_stop()
        coord.join(threads)
    
if __name__ == '__main__':
    main()
