import os

import numpy as np
import tensorflow as tf


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_binarized_vector_with_bcgd(catg_file, num_classes):
    """
    Generates binarized category vector from category info file with background class.
    """
    f = open(catg_file, 'r')
    binary_catg_vec = np.zeros((num_classes)).astype(np.float32)
    binary_catg_vec[0] = 1 # Set background
    for catg in f:
        binary_catg_vec[int(catg)] = 1

    return binary_catg_vec

def get_binarized_vector_wo_bcgd(catg_file, num_classes):
    """
    Generates binarized category vector from category info file without background class.
    """
    f = open(catg_file, 'r')
    binary_catg_vec = np.zeros((num_classes - 1)).astype(np.float32)
    for catg in f:
        binary_catg_vec[int(catg)-1] = 1

    return binary_catg_vec

def image_scaling(img, seed):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      seed: Random seed.
    """

    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=seed)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)

    return img

def image_mirroring(img, seed):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      seed: Random seed.
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32, seed=seed)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    return img

def random_crop_and_pad_image_and_labels(image, crop_h, crop_w, seed):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      seed: Random seed.
    """

    image_shape = tf.shape(image)
    img_pad = tf.image.pad_to_bounding_box(image, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))

    img_crop = tf.random_crop(img_pad, [crop_h,crop_w,3], seed=seed)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    return img_crop

def read_labeled_image_forward_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truths.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form
      '/path/to/image /path/to/image-labels'.
       
    Returns:
      Two lists with all file names for images and image-labels, respectively.
    """

    f = open(data_list, 'r')
    images = []
    catgs = []
    for line in f:
        image, catg = line.strip("\n").split(' ')
        images.append(data_dir + image)
        catgs.append(data_dir + catg)
    return images, catgs

def read_images_from_disk_forward(input_queue, input_size, random_scale, random_mirror, seed, num_classes):
    """Read one image and its corresponding binarized category vector.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      seed: Random seed.
      num_classes: Total number of classes (including background) in the dataset.
      
    Returns:
      Three tensors: the decoded image, approximate ground-truth mask and binarized category vector.
    """

    img_contents = tf.read_file(input_queue[0])

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    catg_with_bcgd = tf.py_func(get_binarized_vector_with_bcgd, [input_queue[1], num_classes], [tf.float32])
    catg_with_bcgd = tf.to_float(tf.reshape(catg_with_bcgd, [num_classes]))
    catg_with_bcgd.set_shape((num_classes))

    catg_wo_bcgd = tf.py_func(get_binarized_vector_wo_bcgd, [input_queue[1], num_classes], [tf.float32])
    catg_wo_bcgd = tf.to_float(tf.reshape(catg_wo_bcgd, [num_classes-1]))
    catg_wo_bcgd.set_shape((num_classes-1))

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img = image_scaling(img, seed)

        # Randomly mirror the images and labels.
        if random_mirror:
            img = image_mirroring(img, seed)

        # Randomly crops the images and labels.
        img = random_crop_and_pad_image_and_labels(img, h, w, seed)

    return img, catg_with_bcgd, catg_wo_bcgd


class ImageReader_Classfc(object):
    '''Generic ImageReader which reads images, attentions, saliency and category
       information from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, seed, random_scale, random_mirror, 
                 num_classes, coord):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form
                '/path/to/image /path/to/attn /path/to/sal /path/to/image-labels'
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          seed: Random seed.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          num_classes: Total number of classes (including background) in the dataset.
          coord: TensorFlow queue coordinator.
        '''

        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.seed = seed
        self.coord = coord

        self.image_list, self.catg_list = read_labeled_image_forward_list(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.catgs = tf.convert_to_tensor(self.catg_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.catgs],
                                                   shuffle=input_size is not None)
        self.image, self.catg_with_bcgd, self.catg_wo_bcgd = read_images_from_disk_forward(self.queue, self.input_size,\
                                                              random_scale, random_mirror, self.seed, num_classes)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Three tensors of for images approximate ground-truth masks and binarized category vectors.'''

        image_batch, catg_with_bcgd_batch, catg_wo_bcgd_batch = tf.train.batch\
                ([self.image, self.catg_with_bcgd, self.catg_wo_bcgd], num_elements)
        return image_batch, catg_with_bcgd_batch, catg_wo_bcgd_batch
