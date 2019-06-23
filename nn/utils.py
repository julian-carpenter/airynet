import tensorflow as tf
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
import numpy as np
import h5py
import cv2
import os
from nn import get_config

cfg, _ = get_config()
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def log_act(inputs, positive_flag=True):
    """
    This is a custom activation using two log functions
    Symmetrical with respect to the origin (zero).
    """
    exp_neg_one = tf.exp(tf.constant(-1.), name="exp_neg_one")
    if positive_flag:
        alpha = cfg.alpha
    else:
        alpha = -cfg.alpha
        inputs = -inputs
    return alpha * (tf.log(inputs + exp_neg_one) + 1) + cfg.beta


def batch_norm_act_fun(inputs,
                       is_training,
                       data_format,
                       relu_leakiness,
                       reuse,
                       use_bn=True):
    """
    Performs a batch normalization followed by an activation.
    """
    if use_bn:
        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=1 if data_format == "channels_first" else 3,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=is_training,
            fused=True,
            reuse=reuse)

    if relu_leakiness > 0:
        inputs = tf.nn.leaky_relu(inputs, alpha=relu_leakiness)
    else:
        inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size, data_format):
    """
    Pads the input along the spatial dimensions independently of input size.

      Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
          [batch, height_in, width_in, channels] depending on data_format.
        kernel_size: The kernel to be used in the conv2d or max_pool2d
                     operation. Should be a positive integer.
        data_format: The input format ("channels_last" or "channels_first").

      Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == "channels_first":
        padded_inputs = tf.pad(
            inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(
            inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format,
                         reuse,
                         activation=None,
                         name=None):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=("SAME" if strides == 1 else "VALID"),
        activation=activation,
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(
            scale_l1=cfg.l1_regularization_scale,
            scale_l2=cfg.l2_regularization_scale),
        data_format=data_format,
        reuse=reuse,
        name=name)


class PredictionHook(session_run_hook.SessionRunHook):
    """
    This custom hook does two things:
    1) It runs the gradient cam.
        Following https://thehive.ai/blog/inside-a-neural-networks-mind

        Based on:
        Chattopadhyay, A., Sarkar, A., Howlader, P. (2017)
        Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep
        Convolutional Networks. Retrieved from http://arxiv.org/abs/1710.11063
    2) It saves the GradCam Images, the Original Images, the bid, the predicted
        classes and the prediction probabilities into a large h5 file.
    """

    def __init__(self, out_path=None):
        """
        Initializing

        out_path = save path for the hdf5 file
        """
        if not out_path:
            self.h5file = os.path.join(os.path.curdir, "predictions",
                                       "{}.h5".format(cfg.dataset))
        else:
            self.h5file = out_path

        if os.path.isfile(self.h5file):
            os.remove(self.h5file)
        if cfg.dataset == "cxidb":
            self.counter = 0

    def begin(self):
        # Get the Graph.
        self.graph = tf.get_default_graph()

        self.requests = {
            "bids":
            self.graph.get_operation_by_name("bids").outputs[0],
            "images":
            self.graph.get_operation_by_name("images").outputs[0],
            "preds":
            self.graph.get_operation_by_name(
                "sigmoid_output_classes").outputs[0],
            "probs":
            self.graph.get_operation_by_name("sigmoid_output").outputs[0],
            "conv":
            self.graph.get_operation_by_name(
                "device_GPU_0/fourth_block/last_block_before_fc").outputs[0],
            "grads":
            self.graph.get_operation_by_name("normalized_grads").outputs[0]
        }

    def before_run(self, run_context):
        return SessionRunArgs(self.requests)

    def after_run(self, run_context, run_values):
        """
        For every batch, calculate GradCam, get the preds and probs and
        save everything to h5.
        """
        with h5py.File(self.h5file, "a", libver="latest") as fid:
            for idx in range(np.shape(run_values.results["conv"])[0]):
                out_ = run_values.results["conv"][idx]
                bids_ = run_values.results["bids"][idx]
                images_ = run_values.results["images"][idx]
                preds_ = run_values.results["preds"][idx]
                probs_ = run_values.results["probs"][idx]

                if cfg.dataset == "cxidb":
                    grp = fid.create_group("{}".format(np.squeeze(
                        self.counter)))
                    grp.create_dataset("ground_truth",
                                       data=np.squeeze(bids_),
                                       dtype="f")
                    grp.create_dataset(
                        "pred",
                        data=np.argmax(probs_).astype(int).squeeze(),
                        dtype="i")

                    self.counter += 1
                else:
                    grp = fid.create_group("{}".format(np.squeeze(bids_)))

                for class_idx in range(cfg.num_classes):
                    grads_ = run_values.results["grads"][class_idx, idx]
                    # print(np.shape(grads_))
                    weights = np.mean(grads_, axis=(-2, -1))
                    # print(np.shape(weights))
                    cam = np.ones(run_values.results["conv"].shape[-2:],
                                  dtype=np.float32)
                    # print(np.shape(cam))
                    # Taking a weighted average
                    for i, w in enumerate(weights):
                        cam += w * out_[i, :, :]

                    # Passing through ReLU
                    cam = np.maximum(cam, 0)
                    cam = cam / np.max(cam)
                    grad_cam_img = cv2.resize(
                        cam, (cfg.target_width, cfg.target_width))

                    class_group = grp.create_group("{}".format(class_idx))
                    class_group.create_dataset("grad_cam",
                                               data=np.squeeze(grad_cam_img),
                                               dtype="f")

                grp.create_dataset("img", data=images_.squeeze(), dtype="f")
                grp.create_dataset("prediction",
                                   data=np.round(preds_).squeeze(),
                                   dtype="i")
                grp.create_dataset("probability",
                                   data=probs_.squeeze(),
                                   dtype="f")
