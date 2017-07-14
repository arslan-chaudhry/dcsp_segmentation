import os

import numpy as np
import tensorflow as tf

EPSILON = 1e-12
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_label_shape(label):
    """
    Returns the shape of a tensor
    """
    shape = np.asarray(label.shape)
    return shape

def get_binarized_label_vector(catg_file, n_classes):
    """
    Generates binarized category vector from category info file.
    """
    f = open(catg_file, 'r')
    binary_catg_vec = np.zeros((n_classes-1)).astype(np.float32)
    for catg in f:
        binary_catg_vec[int(catg) - 1] = 1

    return binary_catg_vec

def get_multiclass_labels(catg_file):
    """
    Reads class labels from category info file.
    """
    f = open(catg_file, 'r')
    mapIDs = []
    mapIDs.append(0)
    for l in f:
        mapIDs.append(l.rstrip('\n'))

    catgs = np.zeros((len(mapIDs))).astype(np.int32)

    for i in range(len(mapIDs)):
        catgs[i] = int(mapIDs[i])

    return catgs

def get_localization_cues(attn_file, saliency_file, catg_file, n_classes, adapt):
    """
    Generates the localization cues.

    Args:
        attn_file: Path to file containing attention.
        saliency_file: Path to file containing saliency.
        catg_file: Path to file containing category information.
        n_classes: Total number of classes (including background) in the dataset.
        adapt: Whether to obtain adapted ground truth cues (True/ False).
    """

    # Load the npz files
    attn_arr = np.load(attn_file)

    if adapt:
        img_label = attn_arr['actv'].astype(np.int32)
    else:
        # Extract attention and saliency
        attn = attn_arr['actv'].astype(np.float64)
        saliency_arr = np.load(saliency_file)
        saliency = saliency_arr['actv'].astype(np.float64)

        # Get the categories present
        catg_IDs = get_multiclass_labels(catg_file)

        # Append a channel dimension to saliency
        saliency = saliency.reshape((saliency.shape[0], saliency.shape[1], 1))

        # Calculate harmonic mean
        hm = 2 / ((1/(attn + EPSILON)) + (1/(saliency + EPSILON)))
        maxs = np.amax(hm, axis=2)
        bckg_prob = np.zeros((maxs.shape[0], maxs.shape[1]))
        bckg_prob [maxs < 0.4] = 1
        maxs = np.reshape(maxs, ((maxs.shape[0], maxs.shape[1], 1)))
        bckg_prob = np.reshape(bckg_prob, ((bckg_prob.shape[0], bckg_prob.shape[1], 1)))
        hm_max = np.equal(hm, maxs).astype(np.int32) * hm
        concat_hm = np.concatenate((bckg_prob, hm_max), axis=2)

        # Get the final map
        final_label = np.zeros((saliency.shape[0], saliency.shape[1], n_classes)).astype(np.float64)
        final_label[:, :, catg_IDs] = concat_hm
        img_label = np.argmax(final_label, axis=2).astype(np.int32)
        img_label = np.reshape(img_label, ((final_label.shape[0], final_label.shape[1], 1))).astype(np.int32)

    return img_label

def image_scaling(img, label, seed):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
      seed: Random seed.
    """
    
    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=seed)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
   
    return img, label

def image_mirroring(img, label, seed):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
      seed: Random seed.
    """
    
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32, seed=seed)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label

def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, seed):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      seed: Random seed.
    """
    label = tf.cast(label, dtype=tf.float32)
    combined = tf.concat(axis=2, values=[image, label]) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),\
                                                tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4], seed=seed)
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = tf.cast(label_crop, dtype=tf.int32)
    
    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop  

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truths.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form
      '/path/to/image /path/to/attn /path/to/sal /path/to/image-labels'.
       
    Returns:
      Four lists with all file names for images, attention, saliencies and image-labels, respectively.
    """
    f = open(data_list, 'r')
    images = []
    attns = []
    sals = []
    catgs = []
    for line in f:
        image, attn, sal, catg = line.strip("\n").split(' ')
        images.append(data_dir + image)
        attns.append(data_dir + attn)
        sals.append(data_dir + sal)
        catgs.append(data_dir + catg)
    return images, attns, sals, catgs

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, seed, n_classes, adapt):
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      seed: Random seed.
      n_classes: Total number of classes (including background) in the dataset.
      adapt: Whether to obtain adapted ground truth cues (True/ False).
      
    Returns:
      Three tensors: the decoded image, approximate ground-truth mask and binarized category vector.
    """

    img_contents = tf.read_file(input_queue[0])
    
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    label = tf.py_func(get_localization_cues, \
                       [input_queue[1], input_queue[2], input_queue[3], n_classes, adapt], [tf.int32])
    shape = tf.py_func(get_label_shape, label, [tf.int64])
    shape = tf.to_int32(tf.reshape(shape, [3]))
    label = tf.to_int32(tf.reshape(label, shape))

    catg = tf.py_func(get_binarized_label_vector, [input_queue[3], n_classes], [tf.float32])
    catg = tf.to_float(tf.reshape(catg, [n_classes-1]))
    catg.set_shape((n_classes-1))

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, label = image_scaling(img, label, seed)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label = image_mirroring(img, label, seed)

        # Randomly crops the images and labels.
        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, seed)

    return img, label, catg

class ImageReader_Segment(object):
    '''Generic ImageReader which reads images, attentions, saliency and category
       information from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, seed, random_scale,
                 random_mirror, num_classes, adapt, coord):
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
          adapt: Whether to obtain adapted ground truth cues (True/ False).
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.seed = seed
        self.coord = coord
        
        self.image_list, self.attn_list, self.sal_list, self.catg_list = read_labeled_image_list\
                (self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.attns = tf.convert_to_tensor(self.attn_list, dtype=tf.string)
        self.saliencys = tf.convert_to_tensor(self.sal_list, dtype=tf.string)
        self.catgs = tf.convert_to_tensor(self.catg_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.attns, self.saliencys, self.catgs],
                                                   shuffle=input_size is not None) # not shuffling if it is val
        self.image, self.label, self.catg = read_images_from_disk\
                (self.queue, self.input_size, random_scale, random_mirror, self.seed, num_classes, adapt) 

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Three tensors of for images approximate ground-truth masks and binarized category vectors.'''
        image_batch, label_batch, catg_batch = tf.train.batch([self.image, self.label, self.catg],
                                                  num_elements)
        return image_batch, label_batch, catg_batch
