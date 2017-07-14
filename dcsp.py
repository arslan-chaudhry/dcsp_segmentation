"""Training script with multi-scale inputs for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import shutil

# import modules from different scripts
import get_localization
import train_fcan
import train_segmentation

# Defaults of commandline variables
DATA_DIRECTORY = '/home/VOCdevkit'
CLASSFC_DATA_LIST = './dataset/train_labels_only.txt'
SEGMENT_DATA_LIST = './dataset/train_attn_sal_labels.txt'
RESTORE_FROM = './deeplab_resnet.ckpt'
SNAPSHOT_DIR = './snapshots/'
ATTN_SNAPSHOT_DIR = './attn_snapshots'

DATASET_NAME = 'PASCAL_VOC_Aug'
INPUT_SIZE = '321,321'
NUM_CLASSES=21

RANDOM_SEED = 1234
ADAPT_AFTER = 10000
CLASSFC_STEPS = 30000
ADAPT_STEPS = 1
LEARNING_RATE = 1e-3

ATTENUATE_LR_BY = 0.7
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Discovering Class Specific Pixels (DCSP) Network.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--classfc-data-list", type=str, default=CLASSFC_DATA_LIST,
                        help="Path to the list containing images and image level labels.")
    parser.add_argument("--segment-data-list", type=str, default=SEGMENT_DATA_LIST,
                        help="Path to the list containing images, attention, saliency and image level labels.\
                        Note: Script will create the directory for storing attentions in the --data-dir.")
    parser.add_argument("--classfc-steps", type=int, default=CLASSFC_STEPS,
                        help="Number of training steps for which the attention network is trained.")
    parser.add_argument("--adapt-after", type=int, default=ADAPT_AFTER,
                        help="Number of training steps after which attentions are updated.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay. Note: The same learning rate\
                         will be used to train both the attention and segmentation networks.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="From where to restore model parameters for attention network. Note: Segmentation network\
                        is seeded by the weights from the classification/ attention network.")
    parser.add_argument("--attn-snapshot-dir", type=str, default=ATTN_SNAPSHOT_DIR,
                        help="Where to save snapshots of the attention network.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the segmentation network.")
    return parser.parse_args()

def main():
    """Main function to run EM algorithm.

    Output:
    """
    # Get the command-line arguments
    args = get_arguments()

    # Get input crop size
    h, w = map(int, INPUT_SIZE.split(','))
    input_size = (h, w)

    attn_save_dir = args.data_dir.rstrip('\/') + '/' + DATASET_NAME + '/' + 'Attentions/'

    # Create the directory to store attentions
    if not os.path.exists(attn_save_dir):
        os.makedirs(attn_save_dir)

    # Train the attention network
    print('\n\nStarted training the attention network ...\n\n')
    train_fcan.main(data_dir=args.data_dir, data_list=args.classfc_data_list, start_step=0,\
                    num_steps=args.classfc_steps+1, restore_from=args.restore_from,\
                    snapshot_dir=args.attn_snapshot_dir, base_learning_rate=args.learning_rate,\
                    n_classes=NUM_CLASSES, input_size=input_size)

    attn_restore_from = args.attn_snapshot_dir.rstrip('\/') + '/' + 'model.ckpt-%d'%(args.classfc_steps)
    for i in range(ADAPT_STEPS+1):

        if(i == 0):
            # Get the approximate ground-truth cues before adapt
            print('\n\nStoring the initial localization cues in {}...\n\n'.format(attn_save_dir))
            get_localization.main(data_dir=args.data_dir, data_list=args.classfc_data_list,\
                                   restore_from=attn_restore_from, save_dir=attn_save_dir,\
                                   n_classes=NUM_CLASSES, adapt=False)

            # Train the network for segmentation before adapt
            print('\n\nStarted training the segmentation network before adapt ...\n\n')
            train_segmentation.main(data_dir=args.data_dir, data_list=args.segment_data_list, start_step=0,\
                                    num_steps=args.adapt_after+1, global_step=i*args.adapt_after,\
                                    restore_from=attn_restore_from, snapshot_dir=args.snapshot_dir,\
                                    base_learning_rate=args.learning_rate, n_classes=NUM_CLASSES, adapt=False,\
                                    input_size=input_size)

        else:
            # Adapt the localization cues
            print('\n\nStoring the localization cues after adapt in {}...\n\n'.format(attn_save_dir))
            attn_restore_from = args.snapshot_dir.rstrip('\/') + '/' + 'model.ckpt-%d'%(i*args.adapt_after)
            get_localization.main(data_dir=args.data_dir, data_list=args.classfc_data_list,
                                   restore_from=attn_restore_from, save_dir=attn_save_dir,\
                                   n_classes=NUM_CLASSES, adapt=True)

            # Train the network for segmentation after adapt
            print('\n\nStarted training the segmentation network after adapt ...\n\n')
            train_segmentation.main(data_dir=args.data_dir, data_list=args.segment_data_list, start_step=0,\
                                    num_steps=args.adapt_after+1, global_step=i*args.adapt_after,\
                                    restore_from=attn_restore_from, snapshot_dir=args.snapshot_dir,\
                                    base_learning_rate=args.learning_rate, n_classes=NUM_CLASSES, adapt=True,\
                                    input_size=input_size)

if __name__ == '__main__':
    main()

