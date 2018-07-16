# Using deep neural nets for classification of diffraction images

## Description

Code for an upcoming paper; "Using deep neural nets for classification of diffraction images", Zimmermann.

**Not yet released and not yet ready for deployment.**


## Abstract

Intense short-wavelength pulses from Free-Electron-Lasers and high-harmonic-generation sources enable diffractive imaging of individual nano-sized objects with a single x-ray laser shot. Due to the high repetition rates, large data sets with up to several million diffraction patterns are typically obtained in Free-Electron-Laser particle diffraction experiments, representing a severe problem for data analysis. Assuming a dataset of K diffraction patterns with M x N pixels, a high dimensional space (K x M x N) has to be analyzed. Thus feature selection is crucial as it reduces the dimensionality. This is typically achieved via custom-made algorithms that do not generalize well, such as feature extraction methods applicable to spherically shaped patterns but not to arbitrary shapes. We exploit the possibility to utilize a deep neural network as a feature extractor. We benchmark two widely used deep neural network architectures, a residual convolutional deep neural network - called ResNet and a classical convolutional neural network - called VGG -, both trained in a supervised manner utilizing a small training set of manually labeled data. The classification accuracy of these models is enhanced by implementing a novel activation function, which takes the intrinsic scaling of diffraction pattern into account. Furthermore, an approach to be more robust to highly noisy data using two-point cross-correlation maps is presented. The approaches are tested and benchmarked on two large datasets. The first dataset is from a wide-angle X-ray scattering experiment on Helium nanodroplets, conducted at the LDM endstation of the FERMI free-electron laser in Trieste, and the second dataset is from a small-angle X-ray scattering dataset from the CXI-database. Our results show that deep convolutional neural networks are able to outperform previous attempts for sorting and classifying complex diffraction pattern.

## Requirements
* Python 3.6+ (Mainly because of the _print("...{}...".format(...)))_ statements)
* _Tensorflow_ 1.4+ (Only tested on 1.7+ though)
* The following Python packages: _os_, _glob_, _datetime_, _tqdm_, _argparse_, _cv2_, _h5py_ and _numpy_

## Usage
```python
python airynet.py ––flags=value

Possible flags: # declared in _nn/config.py_

# Network
"--resnet_size",
    type=int,
    help="The size of the resnet model to use. Only applicable to resnet")
"--vgg_size",
    type=int,
    help="The size of the vgg model to use. Only applicable to vgg")
"--airynet_type",
    type=str,
    help="The architecture that is used")

# Data
"--data_dir",
    type=str,
    help="This path is a prefix for the dataset flag.")
"--dataset",
    type=str,
    help="Name of the dataset. Folder in cfg.data_dir/cfg.dataset")
"--log_dir",
    type=str,
    help="The directory where the model will be stored.")
"--load_dir",
    type=str,
    help="Restore the network graph from there. Leave blank otherwise")
"--save_dir",
    type=str,
    help="The network graph will be saved to this directory. When mode=save")
"--out_dir_name",
    type=str,
    help="If provided this tag is used as output folder name for the model")
"--mode",
    type=str,
    help="Which mode is currently active")
"--batch_size",
    type=int,
    help="The number of images per batch.")
"--dataset_size",
    type=int,
    help="The number of images in the dataset")
"--ori_width",
    type=int,
    help="Original width of the input images")
"--ori_height",
    type=int,
    help="Original height of the input images")
"--ori_depth",
    type=int,
    help="Original depth (channels) of the input images")
"--target_width",
    type=int,
    help="Target width of the input images")
"--target_height",
    type=int,
    help="Target height of the input images")
"--target_depth",
    type=int,
    help="Target depth (channels) of the input images")
"--num_classes",
    type=int,
    help="The number classes for classification")
"--num_worker",
    type=int,
    help="Number of simultaneous threads that read in data")

# Training
"--train_steps",
    type=int,
    help="The number of batches to train.")
"--steps_per_eval",
    type=int,
    help="The number of batches to run in between evaluations.")
"--lr",
    type=float,
    help="lr is scaled with batch size. lr_final = lr*batch_size/128")
"--gamma",
    type=float,
    help="The momentum (gamma) for the optimizer")
"--l1_regularization_scale",
    type=float,
    help="Apply l1 regularization")
"--l2_regularization_scale",
    type=float,
    help="Apply l2 regularization")
"--use_weight_decay",
    type=str2bool,
    help="Use a l2 weight decay loss")
"--weight_decay",
    type=float,
    help="The amount of weight decay")
"--relu_leakiness",
    type=float,
    help="leakiness of relu activations. only when relu is used")
"--alpha",
    type=float,
    help="alpha value for the log activation")
"--use_log_act",
    type=str2bool,
    help="use the logarithmic activation functions in the first conv layer")
"--use_nchw",
    type=str2bool,
    help="switch gpu tensor from nhwc to nchw")

# Misc.
"--log_level",
    type=str,
    help="The log level for tensorflow")
"--log_step",
    type=int,
    help="Save a chkpnt every log_step steps")
"--random_seed",
    type=int,
    help="Use the same random seed for reproducibility")


```
