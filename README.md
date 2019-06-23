# Deep neural networks for classifying complex features in diffraction images

## Description

Code for : "Deep neural networks for classifying complex features in diffraction images", Zimmermann et al (2019) (Phys. Rev. E 99, 063309)

**https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.063309**


## Abstract

Intense short-wavelength pulses from free-electron lasers and high-harmonic-generation sources enable diffractive imaging of individual nano-sized objects with a single x-ray laser shot. The enormous data sets with up to several million diffraction patterns represent a severe problem for data analysis, due to the high dimensionality of imaging data. Feature recognition and selection is a crucial step to reduce the dimensionality. Usually, custom-made algorithms are developed at a considerable effort to approximate the particular features connected to an individual specimen, but facing different experimental conditions, these approaches do not generalize well. On the other hand, deep neural networks are the principal instrument for today's revolution in automated image recognition, a development that has not been adapted to its full potential for data analysis in science. We recently published in [Langbehn et al. Phys. Rev. Lett. 121, 255301 (2018)] the first application of a deep neural network as a feature extractor for wide-angle diffraction images of helium nanodroplets. Here we present the setup, our modifications and the training process of the deep neural network for diffraction image classification and its systematic benchmarking. We find that deep neural networks significantly outperform previous attempts for sorting and classifying complex diffraction patterns and are a significant improvement for the much-needed assistance during post-processing of large amounts of experimental coherent diffraction imaging data.

## Requirements
* Python 3.6+ (Mainly because of the _print("...{}...".format(...)))_ statements)
* _Tensorflow_ 1.7+
* The following Python packages: _os_, _glob_, _datetime_, _tqdm_, _argparse_, _cv2_, _h5py_ and _numpy_

### WE HIGHLY RECOMMEND TO WRITE YOUR OWN DATA-INPUT ROUTINE, nn/data.py IS HIGHLY SPECIFIC TO THE DATA WE USED IN OUR EXPERIMENTS.

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

## GradCam++ Output
The Prediction hook defined in nn/utils.py generates a large h5 File that includes:
* bids (Obtained from the _label_ Protobuf field)
* images (The original image)
* preds (The predictions -> round(logit(x)))
* probs (Probabilities -> Sigmoid output)
* conv (Filter of the last convolutional layer)
* grads (GradCam++ Gradients)
