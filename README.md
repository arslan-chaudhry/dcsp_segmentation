# DCSP: Discovering Class Specific Pixels for Weakly Supervised Semantic Segmentation

This is the implementation of the [DCSP paper](https://arxiv.org/abs/1707.05821) in tensorflow. The network architecture is inspired from the DeepLab-v2 model, and the [tensorflow reimplementation](https://github.com/DrSleep/tensorflow-deeplab-resnet) of the DeepLab-v2 is used. 

When using this code, please cite our paper:

    @article{chaudhry_dcsp_2017,
      title={Discovering Class-Specific Pixels for Weakly-Supervised Semantic Segmentation},
      author={Arslan Chaudhry and Puneet K. Dokania and Philip H. S. Torr},
      journal={British Machine Vision Conference (BMVC)},
      year={2017}
    }

## Model Description

Please refer to our [paper](https://arxiv.org/abs/1707.05821) for the full details of our model. The attention and segmentation models are based on DeepLab-v2 with an additional convolutional layer to obtain fully convolutional attention maps. For saliency cues, we used [DHSNet](https://drive.google.com/file/d/0B1sbejbIJIW3RlJJY1NNNkFydEU/view) as an off-the-shelf saliency detector. One can use saliency detector of one's choice. Please cite the respective work when using their code. We use our Hierarchical Saliency algorithm to improve the saliency maps (refer to our paper for details). The code in this repository assumes that saliency maps have already been obtained off-line prior to running the scripts in this repository. 


## Requirements

TensorFlow needs to be installed before running the scripts.
TensorFlow v1.1.0 is supported.

To install the required python packages (except TensorFlow), run
```bash
pip install -r requirements.txt
```
or for a local installation
```bash
pip install -user -r requirements.txt
```

## Initialization Models

The initialization and pretrained models are provided [here](https://www.dropbox.com/sh/po12l7zrrf08l4g/AADOsCh0Gb-mJ1fnwSbE7jIBa?dl=0). The model uses the ImageNet initialization of ResNet-101, except for the last `fc*` layers where the weights are initialized by the gaussian with `0` mean and `0.01` standard deviation, and biases with `0`. 

One can use the initialization model provided [above](https://www.dropbox.com/sh/po12l7zrrf08l4g/AADOsCh0Gb-mJ1fnwSbE7jIBa?dl=0) or can convert the vanilla `.caffemodel` of ResNet-101. To convert the initialization model from the `caffemodel`, download the appropriate `.caffemodel` file, and install [Caffe to TensorFlow](https://github.com/ethereon/caffe-tensorflow) dependencies. The Caffe model definition is provided in `misc/deploy.prototxt`. 
To extract weights from `.caffemodel`, run the following:
```bash
python convert.py /path/to/deploy/prototxt --caffemodel /path/to/caffemodel --data-output-path /where/to/save/numpy/weights
```
As a result of running the command above, the model weights will be stored in `/where/to/save/numpy/weights`. To convert them to the native TensorFlow format (`.ckpt`), simply execute:
```bash
python npy2ckpt.py /where/to/save/numpy/weights --save-dir=/where/to/save/ckpt/weights
```

## Dataset and Training

To train the network, one can use the augmented PASCAL VOC 2012 dataset with `10582` images for training and `1449` images for validation. We do not make use of pixel-level annotations of the training/ validation sets and only use the image tags. 

Prepare the dataset by extracting the JPEG images and image tags from the PASCAL_VOC2012 dataset in a directory. Please consult the list `dataset/train_labels_only.txt` in the repository for the names of the directories and corresponding files. For ease of use, extracted image tags are provided [here](https://www.dropbox.com/sh/po12l7zrrf08l4g/AADOsCh0Gb-mJ1fnwSbE7jIBa?dl=0). Save the saliency masks (in the `Saliency` folder) at the same level where JPEGImages and ImageTags are stored. Please see the list `dataset/train_attn_sal_labels.txt` in the repository for reference. Note that we have used [DHSNet](https://drive.google.com/file/d/0B1sbejbIJIW3RlJJY1NNNkFydEU/view) to generate saliency masks. A sample `gen_saliency_voc.py` file which uses Hierarchical Saliency algorithm to generate improved saliency masks is added in this repository. Add this file in the [Caffe code](https://drive.google.com/file/d/0B1sbejbIJIW3RlJJY1NNNkFydEU/view) of the DHSNet to generate saliency masks. 

Once the files are extracted as per the lists described above, one can start training the dcsp model. `dcsp.py` is the main script which trains both the attention and segmentation model. To train the model, run the `dcsp.py` script with the appropriate command-line options. To see the documentation on each of the training settings, run the `dcsp.py` script with the `--help` flag:
```bash
python dcsp.py --help
```

One example run of the training script is shown below:

```bash
python dcsp.py --data-dir /home/mac/Downloads/DataSets/ --classfc-data-list ./dataset/train_labels_only.txt --segment-data-list ./dataset/train_attn_sal_labels.txt --classfc-steps 30000 --adapt-after 10000 --restore-from /home/mac/Downloads/resnet_pretrained_classification/model.ckpt --attn-snapshot-dir ./snapshots_attn_network --snapshot-dir ./snapshots_segmentation_network
```

Once the training script exits, the segmentation model is stored in the directory pointed by `--snapshot-dir` option. 

## Evaluation

To evaluate the model, run the `evaluate.py` script with appropriate command line options. The documentation on each of the evaluation option can be obtained by running:
```bash
python evaluate.py --help
```
One example run of the training script is shown below:

```bash
python evaluate.py --data-dir ~/Downloads/DataSets/PASCAL_VOC_Aug/ --data-list ./dataset/val.txt --num-steps 1449 --restore-from ./snapshots_segmentation_network/model.ckpt-20000
```

The CRF post-processing is embedded in the evaluation script. To turn-off the CRF post-processing comment out the lines number 98, 99 from the `evaluate.py` script. 
With CRF post-processing the model gives the mIOU of `60.8%` on PASCAL VOC val set.

## Evaluating on PASCAL Server
A training script `test_voc.py` is provided that allows one to generate segmentation mask in a directory, that can latter be uploaded to the PASCAL VOC evaluation/ test server. Run the `test_voc.py` script with appropriate options to generate the segmentation masks in a directory. The documentation on each of the option can be obtained by running:
```bash
python test_voc.py --help
```

On the test set of PASCAL VOC the model achieves the mIOU of `61.9%`
    
## Questions/ Bugs
* For questions, please contact the author Arslan Chaudhry (arslan.chaudhry@new.ox.ac.uk).
* Please open the bugs against this repository. Any comment/ improvement would be highly appreciated. 
