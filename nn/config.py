"""
The config parameter

Author:
Julian Zimmermann
mail: julian.zimmermann@mbi-berlin.de
github: julian-carpenter
"""
import argparse as ap


def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = ap.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Network
net_arg = add_argument_group("Network")
net_arg.add_argument(
    "--resnet_size",
    type=int,
    default=18,
    choices=[18, 34, 50, 101, 152, 200],
    help="The size of the resnet model to use. Only applicable to resnet")
net_arg.add_argument(
    "--vgg_size",
    type=int,
    default=16,
    choices=[16, 19],
    help="The size of the vgg model to use. Only applicable to vgg")
net_arg.add_argument(
    "--airynet_type",
    type=str,
    default="resnet",
    choices=["resnet", "vgg"],
    help="The architecture that is used")

# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    "--data_dir",
    type=str,
    default="../datasets/",
    help="This path is a prefix for the dataset flag.")
data_arg.add_argument(
    "--dataset",
    type=str,
    default="cxidb",
    help="Name of the dataset. Has to be a folder -> data_dir/dataset")
data_arg.add_argument(
    "--log_dir",
    type=str,
    default="logs",
    help="The directory where the model will be stored.")
data_arg.add_argument(
    "--load_dir",
    type=str,
    default="",
    help="Restore the network graph from there. Leave blank otherwise")
data_arg.add_argument(
    "--save_dir",
    type=str,
    default="",
    help="The network graph will be saved to this directory. When mode=save")
data_arg.add_argument(
    "--out_dir_name",
    type=str,
    default="",
    help="If provided this tag is used as output folder name for the model")
data_arg.add_argument(
    "--mode",
    type=str,
    default="predict",
    choices=["train", "predict"],
    help="Which mode is currently active")
data_arg.add_argument(
    "--batch_size",
    type=int,
    default=24,
    help="The number of images per batch.")
data_arg.add_argument(
    "--dataset_size",
    type=int,
    default=7264,
    help="The number of images in the dataset")
data_arg.add_argument(
    "--ori_width",
    type=int,
    default=259,
    help="Original width of the input images")
data_arg.add_argument(
    "--ori_height",
    type=int,
    default=259,
    help="Original height of the input images")
data_arg.add_argument(
    "--ori_depth",
    type=int,
    default=1,
    help="Original depth (channels) of the input images")
data_arg.add_argument(
    "--target_width",
    type=int,
    default=224,
    help="Target width of the input images")
data_arg.add_argument(
    "--target_height",
    type=int,
    default=224,
    help="Target height of the input images")
data_arg.add_argument(
    "--target_depth",
    type=int,
    default=1,
    help="Target depth (channels) of the input images")
data_arg.add_argument(
    "--num_classes",
    type=int,
    default=5,
    help="The number of classes for classification")
data_arg.add_argument(
    "--num_worker",
    type=int,
    default=4,
    help="Number of simultaneous threads that read in data")

# Training / test parameters
train_arg = add_argument_group("Training")
train_arg.add_argument(
    "--train_steps",
    type=int,
    default=3e4,
    help="The number of batches to train.")
train_arg.add_argument(
    "--steps_per_eval",
    type=int,
    default=2.5e3,
    help="The number of batches to run in between evaluations.")
train_arg.add_argument(
    "--lr",
    type=float,
    default=.1,  # .1 vgg
    help="lr is scaled with batch size. lr_final = lr*batch_size/128")
train_arg.add_argument(
    "--gamma",
    type=float,
    default=0.9,
    help="The momentum (gamma) for the optimizer")
train_arg.add_argument(
    "--l1_regularization_scale",
    type=float,
    default=1e-5,
    help="Apply l1 regularization")
train_arg.add_argument(
    "--l2_regularization_scale",
    type=float,
    default=1e-5,
    help="Apply l2 regularization")
train_arg.add_argument(
    "--use_weight_decay",
    type=str2bool,
    default=False,
    help="Use a l2 weight decay loss")
train_arg.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="The amount of weight decay (only if --use_weight_decay is True)")
train_arg.add_argument(
    "--relu_leakiness",
    type=float,
    default=0.2,
    help="leakiness of relu activations. only when relu is used")
train_arg.add_argument(
    "--use_log_act",
    type=str2bool,
    default=True,
    help="use the logarithmic activation functions in the first conv layer")
train_arg.add_argument(
    "--alpha",
    type=float,
    default=.2,
    help="alpha value for the log activation (only if --use_log_act is True)")
train_arg.add_argument(
    "--beta",
    type=float,
    default=0.,
    help="beta value for the log activation (only if --use_log_act is True)")
train_arg.add_argument(
    "--use_nchw",
    type=str2bool,
    default=True,
    help="switch gpu tensor from nhwc to nchw")

# Misc
misc_arg = add_argument_group("Misc")
misc_arg.add_argument(
    "--log_level",
    type=str,
    default="INFO",
    choices=["INFO", "DEBUG", "WARN"],
    help="The log level for tensorflow")
misc_arg.add_argument(
    "--log_step",
    type=int,
    default=100,
    help="Save a chkpnt every log_step steps")
misc_arg.add_argument(
    "--random_seed",
    type=int,
    default=1612,
    help="Use the same random seed for reproducibility")


def get_config():
    config, unparsed = parser.parse_known_args()
    if config.use_nchw:
        data_format = "channels_first"
    else:
        data_format = "channels_last"
    config.data_format = data_format
    return config, unparsed
