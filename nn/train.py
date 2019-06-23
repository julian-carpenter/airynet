from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nn
import tensorflow as tf
from tensorflow.python.client import device_lib
from numpy import min, max


def model_fn(features, labels, mode, params):
    """Model function for airynet network"""

    def select_architecture(arc="resnet"):
        if arc == "resnet":
            model_fn = nn.airynet_resnet_variant(cfg.resnet_size,
                                                 cfg.num_classes,
                                                 cfg.data_format,
                                                 cfg.relu_leakiness)
        elif arc == "vgg":
            model_fn = nn.airynet_vgg_variant(cfg.vgg_size, cfg.num_classes,
                                              cfg.data_format)
        elif arc == "alexnet":
            model_fn = nn.airynet_alexnet_variant(cfg.num_classes,
                                                  cfg.data_format)
        return model_fn

    if mode == tf.estimator.ModeKeys.PREDICT:
        labels = features["lbl"]
        features = features["imgs"]
        labels = tf.identity(labels, name="bids")
        features = tf.identity(features, name="images")

    feat_converted = tf.map_fn(
        lambda x: tf.image.convert_image_dtype(x, tf.float32), features)
    tf.summary.image("images", feat_converted, max_outputs=3)
    cfg = params["config"]

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        # Multiply the learning rate by 0.1 at 30, 60, 120 and 150 epochs.
        batches_per_epoch = cfg.dataset_size / cfg.batch_size
        boundaries = [
            int(batches_per_epoch * epoch) for epoch in [30, 60, 120, 150]
        ]
        # Scale the learning rate linearly with the batch size. When the
        # batch size is 128, the learning rate should be 0.1.
        lr = cfg.lr * cfg.batch_size / 128
        values = [lr * decay for decay in [1, 0.1, 0.01, 1e-3, 1e-4]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name="learning_rate")
        tf.summary.scalar("learning_rate", learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=cfg.gamma,
                                               use_nesterov=True)

    avail_gpus = get_available_gpus()
    tower_grads = []
    reuse = False

    with tf.variable_scope(tf.get_variable_scope()):
        print(cfg.resnet_size, cfg.num_classes, cfg.data_format,
              cfg.relu_leakiness)
        network = select_architecture(cfg.airynet_type)
        for dev in avail_gpus:
            print("Building inference on: {}".format(dev))

            if int(dev[-1]) != 0:
                # set scope to reuse if more than one gpu are available
                tf.get_variable_scope().reuse_variables()
                reuse = True

            with tf.device(dev), tf.name_scope(
                    dev.replace(":", "_").replace("/", "")):
                logits = network(features, mode == tf.estimator.ModeKeys.TRAIN,
                                 reuse)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    if cfg.dataset == "cifar10":
                        cross_entropy = tf.losses.softmax_cross_entropy(
                            logits=logits, onehot_labels=labels)
                    else:
                        cross_entropy = tf.losses.sigmoid_cross_entropy(
                            logits=logits, multi_class_labels=labels)

                    # get l1_regularizer loss
                    reg_penalty = tf.reduce_mean(
                        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                    # get trainable variables
                    trainable_variables = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES)
                    # get weight decay_loss
                    if cfg.use_weight_decay:
                        loss_weight_decay = tf.reduce_sum(
                            tf.stack([
                                tf.nn.l2_loss(i) for i in trainable_variables
                            ])) * cfg.weight_decay
                    else:
                        loss_weight_decay = 0.

                    # define loss, consider to add the weight_decay
                    loss = cross_entropy + reg_penalty + loss_weight_decay

                    comp_grad_op = optimizer.compute_gradients(
                        loss, trainable_variables)
                    tower_grads.append(comp_grad_op)

        if mode == tf.estimator.ModeKeys.TRAIN:
            grads = average_gradients(tower_grads, tf.get_default_graph())

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Batch norm requires update ops to be
        # added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads, global_step)
    else:
        train_op = None
        loss = tf.constant(0.)

    if cfg.dataset == "cifar10":
        fc_out_activation_fun = tf.nn.softmax
        fc_out_activation_name = "softmax_output"
    else:
        fc_out_activation_fun = tf.nn.sigmoid
        fc_out_activation_name = "sigmoid_output"

    predictions = {
        "classes":
        tf.round(
            fc_out_activation_fun(logits,
                                  name=fc_out_activation_name + "_classes")),
        "probabilities":
        fc_out_activation_fun(logits, name=fc_out_activation_name),
        "bunchID":
        labels
    }
    print(logits.get_shape())

    if mode == tf.estimator.ModeKeys.PREDICT:
        # We calculate the gradients between the output and tha last
        # convolutional layer for the GradCam
        graph = tf.get_default_graph()
        conv_ = graph.get_operation_by_name(
            "device_GPU_0/fourth_block/last_block_before_fc").outputs[0]
        out_ = graph.get_operation_by_name(
            "device_GPU_0/nn_out/final_dense").outputs[0]
        out_ = tf.nn.sigmoid(out_)
        out_ *= tf.round(out_)
        heat_loss_ = []
        grads = []
        norm = []
        normed_grads = []
        for class_idx in range(out_.get_shape()[-1]):
            print("Building GradCam for class: {}".format(class_idx))
            heat_loss_.append(
                tf.reduce_mean(out_[:, class_idx],
                               name="class_loss_{}".format(class_idx)))
            curr_grad = tf.gradients(
                heat_loss_, conv_, name="class_grads_{}".format(class_idx))[0]
            grads.append(curr_grad)
            norm.append(
                tf.sqrt(tf.reduce_mean(tf.square(curr_grad)),
                        name="class_norm_{}".format(class_idx)))
            normed_grads.append(
                tf.divide(tf.convert_to_tensor(grads[class_idx]),
                          tf.convert_to_tensor(norm[class_idx]) +
                          tf.constant(1e-5),
                          name="normalized_grads_{}".format(class_idx)))
        tf.identity(tf.convert_to_tensor(normed_grads),
                    name="normalized_grads")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Create some metrics for logging purposes
    lbl = tf.to_float(labels)
    prediction = predictions["classes"]

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create a tensor named cross_entropy for logging purposes.
        tf.identity(cross_entropy, name="cross_entropy")
        tf.summary.scalar("metrics/cross_entropy", cross_entropy)
        tf.summary.scalar("metrics/reg_penalty", reg_penalty)
        # tf.summary.scalar("metrics/weight_decay_loss", weight_decay_loss)

        # Calculate the confusion matrix
        confusion_matr = tf.to_float(
            tf.confusion_matrix(tf.reshape(lbl, [-1]),
                                tf.reshape(prediction, [-1]),
                                num_classes=2))
        tf.identity(confusion_matr, name="confusion_matr")

        # Matthews Correlation Coefficient
        TP = confusion_matr[1][1]
        TN = confusion_matr[0][0]
        FP = confusion_matr[0][1]
        FN = confusion_matr[1][0]
        MCC = (TP * TN - FP * FN) / (tf.sqrt(
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
        tf.identity(MCC, name="mcc")
        tf.summary.scalar("metrics/mcc", MCC)

        # Stack lbl and predictions as image for the summary
        lbl_vs_prediction = tf.multiply(tf.ones_like(lbl), 255)
        lbl_vs_prediction = tf.expand_dims(tf.stack([
            tf.multiply(lbl_vs_prediction, lbl),
            tf.multiply(lbl_vs_prediction, prediction),
            tf.zeros_like(lbl)
        ],
                                                    axis=-1),
                                           axis=0)
        tf.identity(lbl_vs_prediction, name="lbl_vs_prediction")
        tf.summary.image("metrics/lbl_vs_prediction", lbl_vs_prediction)

        lbl_image = tf.expand_dims(tf.expand_dims(lbl, axis=-1), axis=0)
        tf.identity(lbl_image, name="lbl_image")
        tf.summary.image("metrics/lbl_image", lbl_image)

        prediction_image = tf.expand_dims(tf.expand_dims(prediction, axis=-1),
                                          axis=0)
        tf.identity(prediction_image, name="prediction_image")
        tf.summary.image("metrics/prediction_image", prediction_image)

    accuracy = tf.metrics.accuracy(lbl, prediction)
    tf.identity(accuracy[1], name="train_accuracy")
    tf.summary.scalar("metrics/train_accuracy", accuracy[1])

    eval_tp = tf.metrics.true_positives(lbl, prediction)
    eval_fp = tf.metrics.false_positives(lbl, prediction)
    eval_fn = tf.metrics.false_negatives(lbl, prediction)
    eval_precision = tf.metrics.precision(lbl, prediction)
    eval_mean_per_class = tf.metrics.mean_per_class_accuracy(
        lbl, prediction, cfg.num_classes)
    metrics = {
        "accuracy": accuracy,
        "mean_per_class_accuracy": eval_mean_per_class,
        "precision": eval_precision,
        "true_positives": eval_tp,
        "false_positives": eval_fp,
        "false_negatives": eval_fn
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics)


def get_available_gpus():
    """
    Get a list of available GPU"s
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


def average_gradients(tower_grads, graph):
    """
    Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer
        list is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been
        averaged across all towers.
    """
    with graph.name_scope("averaging_gradients"):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g_idx, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g_idx, 0)

                # Append on a "tower" dimension which we will
                # average over below.
                grads.append(expanded_g)

            # Average over the "tower" dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are
            # shared across towers. So .. we will just return the first
            # tower"s pointer to the Variable.
            vals = grad_and_vars[0][1]
            grad_and_var = (grad, vals)
            average_grads.append(grad_and_var)
        return average_grads


def rescale(vals, new_min, new_max):

    old_min = min(vals)
    old_max = max(vals)

    # range check
    if old_min == old_max:
        print("Warning: Zero input range")
        return vals

    if new_min == new_max:
        print("Warning: Zero output range")
        return vals

    portion = (vals - old_min) * (new_max - new_min) / (old_max - old_min)
    portion = (old_max - vals) * (new_max - new_min) / (old_max - old_min)

    result = portion + new_min
    result = new_max - portion

    return result
