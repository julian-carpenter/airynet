"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation "v2" resnet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation "v2" variant compared to the
"v1" variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nn import get_config, batch_norm_act_fun, conv2d_fixed_padding, log_act
cfg, _ = get_config()


def building_layer(inputs, filters, is_training, projection_shortcut, strides,
                   data_format, relu_leakiness, reuse):
    """
    Standard building block for residual networks with BN before convolutions.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
    (typically a 1x1 convolution when downsampling the input).
    strides: The block"s stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ("channels_last" or "channels_first").

    Returns:
    The output tensor of the block.
    """
    with tf.variable_scope("pre_activation_and_shortcuts"):
        shortcut = inputs
        inputs = batch_norm_act_fun(
            inputs, is_training, data_format, relu_leakiness, reuse=reuse)

        # The projection shortcut should come after the first
        # batch norm and ReLU since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

    with tf.variable_scope("conv_one"):
        inputs = conv2d_fixed_padding(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=strides,
            data_format=data_format,
            reuse=reuse)

    with tf.variable_scope("conv_two"):
        inputs = batch_norm_act_fun(
            inputs, is_training, data_format, relu_leakiness, reuse=reuse)
        inputs = conv2d_fixed_padding(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=1,
            data_format=data_format,
            reuse=reuse)

    return inputs + shortcut


def bottleneck_layer(inputs, filters, is_training, projection_shortcut,
                     strides, data_format, relu_leakiness, reuse):
    """
    Bottleneck block variant for residual networks with BN before convolutions.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that
      the third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
     (typically a 1x1 convolution when downsampling the input).
    strides: The block"s stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ("channels_last" or "channels_first").

    Returns:
    The output tensor of the block.
    """
    with tf.variable_scope("pre_activation_and_shortcuts"):
        shortcut = inputs
        inputs = batch_norm_act_fun(
            inputs, is_training, data_format, relu_leakiness, reuse=reuse)

        # The projection shortcut should come after the first
        # batch norm and ReLU since it performs a 1x1 convolution.

        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)
    with tf.variable_scope("conv_one"):
        inputs = conv2d_fixed_padding(
            inputs=inputs,
            filters=filters,
            kernel_size=1,
            strides=1,
            data_format=data_format,
            reuse=reuse)

    with tf.variable_scope("conv_two"):
        inputs = batch_norm_act_fun(
            inputs, is_training, data_format, relu_leakiness, reuse=reuse)
        inputs = conv2d_fixed_padding(
            inputs=inputs,
            filters=filters,
            kernel_size=3,
            strides=strides,
            data_format=data_format,
            reuse=reuse)

    with tf.variable_scope("conv_three"):
        inputs = batch_norm_act_fun(
            inputs, is_training, data_format, relu_leakiness, reuse=reuse)
        inputs = conv2d_fixed_padding(
            inputs=inputs,
            filters=4 * filters,
            kernel_size=1,
            strides=1,
            data_format=data_format,
            reuse=reuse)

    return inputs + shortcut


def block_layer_fn(inputs, filters, layer_fn, block_layer, strides,
                   is_training, name, data_format, relu_leakiness, reuse):
    """
    Creates one layer of block_layer for the airynet model.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    layer_fn: The block to use within the model, either `building_layer` or
      `bottleneck_layer`.
    block_layer: The number of block_layer contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ("channels_last" or "channels_first").

    Returns:
    The output tensor of the block layer.
    """
    # Bottleneck block_layer end with 4x
    # the number of filters as they start with
    filters_out = 4 * filters if layer_fn is bottleneck_layer else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs,
            filters=filters_out,
            kernel_size=1,
            strides=strides,
            data_format=data_format,
            reuse=reuse)

    # Only the first layer per block uses projection_shortcut and strides
    with tf.variable_scope("layer_0"):
        inputs = layer_fn(inputs, filters, is_training, projection_shortcut,
                          strides, data_format, relu_leakiness, reuse)

    for idx in range(1, block_layer):
        with tf.variable_scope("layer_{}".format(idx)):
            inputs = layer_fn(inputs, filters, is_training, None, 1,
                              data_format, relu_leakiness, reuse)

    return tf.identity(inputs, name)


def airynet_resnet_variant_generator(layer_fn,
                                     layers,
                                     num_classes,
                                     data_format=None,
                                     relu_leakiness=None):
    """
    Generator for ImageNet airynet models.

    Args:
    layer_fn: The block to use within the model, either `building_layer` or
      `bottleneck_layer`.
    layers: A length-4 array denoting the number of block_layer to include in
      each layer. Each layer consists of block_layer that take inputs
      of the same size.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ("channels_last", "channels_first", or None).
      If set to None, the format is dependent on whether a GPU is available.

    Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the airynet model.
    """
    if data_format is None:
        data_format = ("channels_first"
                       if tf.test.is_built_with_cuda() else "channels_last")

    def model(inputs, is_training, reuse):
        """
        Constructs the airynet model given the inputs.
        """
        with tf.variable_scope("nn_in"):
            inp_shape = inputs.get_shape()
            print("Original Image dimensions: {}".format(inp_shape))
            if data_format == "channels_first":
                # Convert from channels_last (channels_last) to
                # channels_first (channels_first).
                # This provides a large performance boost on GPU.
                inputs = tf.transpose(inputs, [0, 3, 1, 2])
                print(
                    "Transpose the inputs to channels_first, from: {} to: {}".
                    format(inp_shape, inputs.get_shape()))
                print("_____________")

            def log_act_helper(x):
                return tf.where(
                    tf.greater(x, 0), log_act(x), log_act(x, False))

            # this is the log layer
            inputs = conv2d_fixed_padding(
                inputs=inputs,
                filters=64,
                kernel_size=7,
                strides=2,
                data_format=data_format,
                reuse=reuse,
                activation=log_act_helper if cfg.use_log_act else None)
            inputs = tf.identity(inputs, "initial_conv")
            print("After the first convolution: {}".format(inputs.get_shape()))
            inputs = tf.layers.max_pooling2d(
                inputs=inputs,
                pool_size=3,
                strides=2,
                padding="SAME",
                data_format=data_format)
            inputs = tf.identity(inputs, "initial_max_pool")
            print("After Max Pooling with pool size three: {}".format(
                inputs.get_shape()))

        with tf.variable_scope("first_block"):
            inputs = block_layer_fn(
                inputs=inputs,
                filters=64,
                layer_fn=layer_fn,
                block_layer=layers[0],
                strides=1,
                is_training=is_training,
                name="first_block_layer_fn",
                data_format=data_format,
                relu_leakiness=relu_leakiness,
                reuse=reuse)
            print("After the first block: {}".format(inputs.get_shape()))

        with tf.variable_scope("second_block"):
            inputs = block_layer_fn(
                inputs=inputs,
                filters=128,
                layer_fn=layer_fn,
                block_layer=layers[1],
                strides=2,
                is_training=is_training,
                name="second_block_layer_fn",
                data_format=data_format,
                relu_leakiness=relu_leakiness,
                reuse=reuse)
            print("After the second block: {}".format(inputs.get_shape()))

        with tf.variable_scope("third_block"):
            inputs = block_layer_fn(
                inputs=inputs,
                filters=256,
                layer_fn=layer_fn,
                block_layer=layers[2],
                strides=2,
                is_training=is_training,
                name="third_block_layer_fn",
                data_format=data_format,
                relu_leakiness=relu_leakiness,
                reuse=reuse)
            print("After the third block: {}".format(inputs.get_shape()))

        with tf.variable_scope("fourth_block"):
            inputs = block_layer_fn(
                inputs=inputs,
                filters=512,
                layer_fn=layer_fn,
                block_layer=layers[3],
                strides=2,
                is_training=is_training,
                name="last_block_before_fc",
                data_format=data_format,
                relu_leakiness=relu_leakiness,
                reuse=reuse)
            print("After the fourth block: {}".format(inputs.get_shape()))

        with tf.variable_scope("nn_out"):
            inputs = batch_norm_act_fun(
                inputs, is_training, data_format, relu_leakiness, reuse=reuse)
            inputs = tf.layers.average_pooling2d(
                inputs=inputs,
                pool_size=7,
                strides=1,
                padding="VALID",
                data_format=data_format)
            inputs = tf.identity(inputs, "final_avg_pool")
            print("After Max Pooling with pool size seven: {}".format(
                inputs.get_shape()))
            inputs = tf.reshape(
                inputs, [-1, 512 if layer_fn is building_layer else 2048])
            print("After reshaping, to serve the fc: {}".format(
                inputs.get_shape()))
            inputs = tf.layers.dense(
                inputs=inputs,
                units=num_classes,
                kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(
                    scale_l1=cfg.l1_regularization_scale,
                    scale_l2=cfg.l2_regularization_scale))
            inputs = tf.identity(inputs, "final_dense")
            print("After final fc layer: {}".format(inputs.get_shape()))
        return inputs

    model.default_image_size = 224
    return model


def airynet_resnet_variant(resnet_size,
                           num_classes,
                           data_format=None,
                           relu_leakiness=None):
    """
    Returns the airynet resnet model for a given size
    and number of output classes.
    """
    model_params = {
        18: {
            "layer_fn": building_layer,
            "layers": [2, 2, 2, 2]
        },
        34: {
            "layer_fn": building_layer,
            "layers": [3, 4, 6, 3]
        },
        50: {
            "layer_fn": bottleneck_layer,
            "layers": [3, 4, 6, 3]
        },
        101: {
            "layer_fn": bottleneck_layer,
            "layers": [3, 4, 23, 3]
        },
        152: {
            "layer_fn": bottleneck_layer,
            "layers": [3, 8, 36, 3]
        },
        200: {
            "layer_fn": bottleneck_layer,
            "layers": [3, 24, 36, 3]
        }
    }

    if resnet_size not in model_params:
        raise ValueError("Not a valid resnet_size:", resnet_size)

    params = model_params[resnet_size]
    print("Building resnet with a {} layout".format(params["layers"]))
    return airynet_resnet_variant_generator(params["layer_fn"],
                                            params["layers"], num_classes,
                                            data_format, relu_leakiness)
