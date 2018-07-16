"""
Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nn import get_config, conv2d_fixed_padding, batch_norm_act_fun, log_act

cfg, _ = get_config()
slim = tf.contrib.slim


def log_act_helper(x):
    return tf.where(tf.greater(x, 0), log_act(x), log_act(x, False))


def vgg_arg_scope(weight_decay=1e-6):
    """
    Defines the VGG arg scope.

    Args:
    weight_decay: The l2 regularization coefficient.

    Returns:
    An arg_scope.
    """
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(weight_decay),
            biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding="SAME") as arg_sc:
            return arg_sc


def conv_act(inputs,
             filters,
             kernel_size=1,
             strides=1,
             leakiness=.1,
             data_format="channels_last",
             reuse=False,
             is_training=True,
             name="",
             scope=""):
    """
    Use global custom conv+activation function
    """
    net = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        data_format=data_format,
        reuse=reuse,
        name=scope + name,
        activation=log_act_helper
        if cfg.use_log_act and cfg.counter == 0 else None)
    net = batch_norm_act_fun(net, is_training, data_format, leakiness, reuse,
                             False)
    if cfg.counter == 0:
        cfg.counter += 1
    return net


def vgg_16(inputs,
           num_classes=11,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope="vgg_16",
           fc_conv_padding="VALID",
           global_pool=True,
           data_format=None,
           reuse=False):
    """
    Oxford Net VGG 16-Layers version D Example.

    Note: All the fully_connected layers have been transformed to
    conv2d layers. To use in classification mode, resize input to 224x224.

    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of
    the outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use "SAME" padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      "VALID" padding.
    global_pool: Optional boolean flag. If True, the input to the
    classification layer is avgpooled to size 1x1, for any input size.
    (This is not part of the original VGG architecture.)

    Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
    """
    if data_format == "channels_first":
        # Convert from channels_last (channels_last) to
        # channels_first (channels_first).
        # This provides a large performance boost on GPU.
        inp_shape = inputs.get_shape()
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        print("Transpose the inputs to channels_first, from: {} to: {}".format(
            inp_shape, inputs.get_shape()))
        print("_____________")
        dat_form_old = "NCHW"
        pool_ = [1, 1, 2, 2]
        global_pool_idx = [2, 3]
    else:
        dat_form_old = "NHWC"
        pool_ = [1, 2, 2, 1]
        global_pool_idx = [1, 2]
    cfg.counter = 0

    with tf.variable_scope(scope, "vgg_16", [inputs]):
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        print("Input Shape: {}".format(inputs.get_shape()))
        net = tf.contrib.layers.repeat(
            inputs,
            2,
            conv_act,
            64,
            3,
            reuse=reuse,
            name="conv1",
            data_format=data_format)
        print("First conv block: {}".format(net.get_shape()))
        net = tf.nn.max_pool(
            net, pool_, pool_, "VALID", name="pool1", data_format=dat_form_old)
        print("After Max-Pooling: {}".format(net.get_shape()))
        net = tf.contrib.layers.repeat(
            net,
            2,
            conv_act,
            128,
            3,
            reuse=reuse,
            name="conv2",
            data_format=data_format)
        print("Second conv block: {}".format(net.get_shape()))
        net = tf.nn.max_pool(
            net, pool_, pool_, "VALID", name="pool2", data_format=dat_form_old)
        print("Max-Pooling: {}".format(net.get_shape()))
        net = tf.contrib.layers.repeat(
            net,
            3,
            conv_act,
            256,
            3,
            reuse=reuse,
            name="conv3",
            data_format=data_format)
        print("Third conv block: {}".format(net.get_shape()))
        net = tf.nn.max_pool(
            net, pool_, pool_, "VALID", name="pool3", data_format=dat_form_old)
        print("Max-Pooling: {}".format(net.get_shape()))
        net = tf.contrib.layers.repeat(
            net,
            3,
            conv_act,
            512,
            3,
            reuse=reuse,
            name="conv4",
            data_format=data_format)
        print("Fourth conv block: {}".format(net.get_shape()))
        net = tf.nn.max_pool(
            net, pool_, pool_, "VALID", name="pool4", data_format=dat_form_old)
        print("Max-Pooling: {}".format(net.get_shape()))
        net = tf.contrib.layers.repeat(
            net,
            3,
            conv_act,
            512,
            3,
            reuse=reuse,
            name="conv5",
            data_format=data_format)
        print("Fifth conv block: {}".format(net.get_shape()))
        net = tf.nn.max_pool(
            net, pool_, pool_, "VALID", name="pool5", data_format=dat_form_old)
        print("Max-Pooling: {}".format(net.get_shape()))

        print("Use conv2d instead of fully_connected layers.")
        net = conv_act(net, 4096, 7, name="fc6", reuse=reuse)
        net = tf.nn.dropout(net, dropout_keep_prob, name="dropout6")
        net = conv_act(net, 4096, 1, name="fc7", reuse=reuse)
        print("Last conv block: {}".format(net.get_shape()))

        if global_pool:
            net = tf.reduce_mean(
                net, global_pool_idx, keepdims=True, name="global_pool")
        if num_classes:
            net = tf.nn.dropout(net, dropout_keep_prob, name="dropout7")
            net = conv2d_fixed_padding(
                net,
                num_classes,
                1,
                1,
                data_format,
                activation=None,
                name="fc8",
                reuse=reuse)
            if spatial_squeeze and num_classes is not None:
                net = tf.squeeze(net, global_pool_idx, name="fc8/squeezed")

        net.default_image_size = 224
        return net


def vgg_19(inputs,
           num_classes=11,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope="vgg_19",
           fc_conv_padding="VALID",
           global_pool=True,
           data_format=None,
           reuse=False):
    """
    Oxford Net VGG 19-Layers version E Example.

    Note: All the fully_connected layers have been transformed to
    conv2d layers. To use in classification mode, resize input to 224x224.

    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of
    the outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use "SAME" padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      "VALID" padding.
    global_pool: Optional boolean flag. If True, the input to the
    classification layer is avgpooled to size 1x1, for any input size.
    (This is not part of the original VGG architecture.)

    Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
    """
    if data_format == "channels_first":
        # Convert from channels_last (channels_last) to
        # channels_first (channels_first).
        # This provides a large performance boost on GPU.
        inp_shape = inputs.get_shape()
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        print("Transpose the inputs to channels_first, from: {} to: {}".format(
            inp_shape, inputs.get_shape()))
        print("_____________")
        dat_form_old = "NCHW"
        pool_ = [1, 1, 2, 2]
        global_pool_idx = [2, 3]
    else:
        dat_form_old = "NHWC"
        pool_ = [1, 2, 2, 1]
        global_pool_idx = [1, 2]
    cfg.counter = 0

    with tf.variable_scope(scope, "vgg_19", [inputs]):
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        print("Input Shape: {}".format(inputs.get_shape()))
        net = tf.contrib.layers.repeat(
            inputs,
            2,
            conv_act,
            64,
            3,
            reuse=reuse,
            name="conv1",
            data_format=data_format)
        print("First conv block: {}".format(net.get_shape()))
        net = tf.nn.max_pool(
            net, pool_, pool_, "VALID", name="pool1", data_format=dat_form_old)
        print("After Max-Pooling: {}".format(net.get_shape()))
        net = tf.contrib.layers.repeat(
            net,
            2,
            conv_act,
            128,
            3,
            reuse=reuse,
            name="conv2",
            data_format=data_format)
        print("Second conv block: {}".format(net.get_shape()))
        net = tf.nn.max_pool(
            net, pool_, pool_, "VALID", name="pool2", data_format=dat_form_old)
        print("Max-Pooling: {}".format(net.get_shape()))
        net = tf.contrib.layers.repeat(
            net,
            4,
            conv_act,
            256,
            3,
            reuse=reuse,
            name="conv3",
            data_format=data_format)
        print("Third conv block: {}".format(net.get_shape()))
        net = tf.nn.max_pool(
            net, pool_, pool_, "VALID", name="pool3", data_format=dat_form_old)
        print("Max-Pooling: {}".format(net.get_shape()))
        net = tf.contrib.layers.repeat(
            net,
            4,
            conv_act,
            512,
            3,
            reuse=reuse,
            name="conv4",
            data_format=data_format)
        print("Fourth conv block: {}".format(net.get_shape()))
        net = tf.nn.max_pool(
            net, pool_, pool_, "VALID", name="pool4", data_format=dat_form_old)
        print("Max-Pooling: {}".format(net.get_shape()))
        net = tf.contrib.layers.repeat(
            net,
            3,
            conv_act,
            512,
            3,
            reuse=reuse,
            name="conv5",
            data_format=data_format)
        print("Fifth conv block: {}".format(net.get_shape()))
        net = tf.nn.max_pool(
            net, pool_, pool_, "VALID", name="pool5", data_format=dat_form_old)
        print("Max-Pooling: {}".format(net.get_shape()))

        print("Use conv2d instead of fully_connected layers.")
        net = conv_act(net, 4096, 7, name="fc6", reuse=reuse)
        net = tf.nn.dropout(net, dropout_keep_prob, name="dropout6")
        net = conv_act(net, 4096, 1, name="fc7", reuse=reuse)
        print("Last conv block: {}".format(net.get_shape()))

        if global_pool:
            net = tf.reduce_mean(
                net, global_pool_idx, keepdims=True, name="global_pool")
        if num_classes:
            net = tf.nn.dropout(net, dropout_keep_prob, name="dropout7")
            net = conv2d_fixed_padding(
                net,
                num_classes,
                1,
                1,
                data_format,
                activation=None,
                name="fc8",
                reuse=reuse)
            if spatial_squeeze and num_classes is not None:
                net = tf.squeeze(net, global_pool_idx, name="fc8/squeezed")

        net.default_image_size = 224
        return net


def airynet_vgg_variant_generator(layers, num_classes, data_format=None):
    """
    Generator for ImageNet airynet v2 models.

    Args:
    layers: A length-1 array denoting the number of layers in the net.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ("channels_last", "channels_first", or None).
      If set to None, the format is dependent on whether a GPU is available.

    Returns:
    The model function that takes in `inputs`, `is_training` and `reuse` and
    returns the output tensor of the airynet model.
    """
    if data_format is None:
        data_format = ("channels_first"
                       if tf.test.is_built_with_cuda() else "channels_last")

    def model(inputs, is_training, reuse):
        """
        Constructs the airynet model given the inputs.
        """
        if layers == 16:
            return vgg_16(
                inputs,
                num_classes=num_classes,
                is_training=is_training,
                data_format=cfg.data_format)
        else:
            return vgg_19(
                inputs,
                num_classes=num_classes,
                is_training=is_training,
                data_format=cfg.data_format)

    model.default_image_size = 224
    return model


def airynet_vgg_variant(vgg_size, num_classes, data_format=None):
    """
    Returns the airynet vgg model for a given size
    and number of output classes.
    """
    if vgg_size not in [16, 19]:
        raise ValueError("Not a valid vgg_size:", vgg_size)

    print("Building: {} layers with a {} layout".format(vgg_size, "VGG"))
    return airynet_vgg_variant_generator(vgg_size, num_classes, data_format)
