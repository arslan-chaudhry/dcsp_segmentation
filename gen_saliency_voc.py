"""
Author: Arslan Chaudhry (arslan.chaudhry@new.ox.ac.uk)
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage import transform, filters
from scipy import misc
import sys, pylab, operator, csv
import os
import urllib
import glob
import util
#%matplotlib inline
from scipy import misc

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

IMG_MEAN = np.array((122.67891434,116.66876762,104.00698793), dtype=np.float32)

import caffe

caffe.set_mode_gpu()
saliency_net = caffe.Net('../models/dhs/SO_RCL_deploy.prototxt',
                '../models/dhs/SO_RCL_models_iter_10000.caffemodel',
                caffe.TRAIN)

imgScale = 224
saliencyTopLayer = 'RCL1_sm'

DATA_DIRECTORY = '/home/mac/VOC'
CATG_PRESENT = 1


def get_arguments():
    parser = argparse.ArgumentParser(description="Attention Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing images.")
    parser.add_argument("--data-list", type=str, default=DATA_DIRECTORY,
                        help="Path to the list containing images.")
    parser.add_argument("--save-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory where masks will be stored.")
    return parser.parse_args()

def resize_image(img_path):
    img = caffe.io.load_image(img_path)
    minDim = min(img.shape[:2])
    newSize = (imgScale, imgScale)
    imgS = transform.resize(img, newSize)
    return img, imgS, newSize

def get_images(data_dir):
    images = []
    for f in glob.iglob(data_dir.rstrip('\/') + '/*.jpg'):
        images.append(f)

    return images

def read_labeled_image_list(data_dir, data_list):
    f = open(data_list, 'r')
    images = []
    for line in f:
        try:
            image = line.strip("\n")
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
    return images

def forward_pass_saliency(imgS, newSize):
    saliency_net.blobs['img'].reshape(1,3,newSize[0],newSize[1])
    transformer = caffe.io.Transformer({'img': saliency_net.blobs['img'].data.shape})
    transformer.set_mean('img', np.array([103.939, 116.779, 123.68]))
    transformer.set_transpose('img', (2,0,1))
    transformer.set_channel_swap('img', (2,1,0))
    transformer.set_raw_scale('img', 255.0)

    saliency_net.blobs['img'].data[...] = transformer.preprocess('img', imgS)
    out = saliency_net.forward(end = saliencyTopLayer)

    msk = saliency_net.blobs[saliencyTopLayer].data[...]
    msk = msk.reshape((imgScale, imgScale))

    return msk

def combined_saliency(final_map, current_map):
    assert(final_map.shape == current_map.shape)
    l, w = final_map.shape
    for i in range(l):
        for j in range(w):
            if(final_map[i, j] > current_map[i, j]):
                final_map[i, j] = final_map[i, j]
            else:
                final_map[i, j] = current_map[i, j]

    return final_map

def replace_with_mean(img, msk):
    l, w = msk.shape
    for i in range(l):
        for j in range(w):
            if msk[i, j] == 0:
                img[i, j, :] = IMG_MEAN/ 255.0

    return img
def main():

    # Get the command line arguments.
    args = get_arguments()

    # Get the filenames
    images = read_labeled_image_list(args.data_dir, args.data_list)


    # Loop through the complete list
    for i in range(len(images)):

        # Load the image
        img = caffe.io.load_image(images[i])

        final_saliency_msk = np.zeros((img.shape[0], img.shape[1]))

        for j in range(2):

            # Resize the image
            minDim = min(img.shape[:2])
            newSize = (imgScale, imgScale)
            imgS = transform.resize(img, newSize)

            # Do the forward pass of saliency
            saliency_msk = forward_pass_saliency(imgS, newSize)

            # Upsize the saliency mask to original size
            saliency_msk = transform.resize(saliency_msk, (img.shape[:2]), order=3, mode='edge')

            final_saliency_msk = combined_saliency(final_saliency_msk, saliency_msk)

            msk = saliency_msk.copy()
            avg = np.mean(msk.flatten())

            if j == 0:
                th = 0.7
            else:
                th = 0.8
            msk[msk >= th] = 1000
            msk[msk < th] = 1
            msk[msk == 1000] = 0
            img = replace_with_mean(img, msk)
           
        base_fname = images[i].strip("\n").rsplit('/', 1)[1].replace('.jpg', '.npz')
        f_name = args.save_dir + "/" + base_fname
        np.savez(f_name, actv=final_saliency_msk)

        """
        # Uncomment this block to see the image visualization
        base_imname = images[i].strip("\n").rsplit('/', 1)[1].replace('.jpg', '.png')
        im_name = args.save_dir + "/" + base_imname
        misc.imsave(im_name, final_saliency_msk)
        """

        if (i % 100 ==0):
            print('Processing category {}/{}'.format(i, len(images)))

if __name__ == '__main__':
    main()
