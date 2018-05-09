from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def get_queue(mode_str, cfg):
    '''
    Defining the loading ops for data management
    '''
    batch_size = cfg.batch_size
    num_worker = cfg.num_worker
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    if cfg.dataset == 'helium' or cfg.dataset == 'cxidb':
        img, lbl = get_dataset(mode_str, cfg)
        if not cfg.use_log_act:
            img = tf.image.convert_image_dtype(img, tf.float32)

        if mode_str == 'predict':
            [img_data, lbl_data] = tf.train.batch(
                [img, lbl],
                batch_size=batch_size,
                num_threads=1,
                capacity=capacity,
                name='input_data')
        else:
            [img_data, lbl_data] = tf.train.shuffle_batch(
                [img, lbl],
                batch_size=batch_size,
                num_threads=num_worker,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                name='input_data')

        if cfg.ori_depth == 1 and cfg.target_depth == 3:
            img_data = tf.image.grayscale_to_rgb(img_data)

        # Crop the image so that is has 2^x dimensions
        crop_height = np.power(2, np.floor(np.log2(cfg.ori_height)))
        crop_width = np.power(2, np.floor(np.log2(cfg.ori_width)))
        img_data = tf.image.resize_image_with_crop_or_pad(
            img_data, crop_height.astype(int), crop_width.astype(int))
        new_interp_size = tf.convert_to_tensor(
            [int(cfg.target_height),
             int(cfg.target_width)], dtype=tf.int32)
        img_data = tf.image.resize_images(img_data, new_interp_size)

        if mode_str == 'train':
            img_data = augment_data(img_data)
        elif mode_str == 'predict' and cfg.dataset == 'helium':
            # if predicting and using the helium dataset
            # the 'labels' field is is filled with the unique
            # bunchID of the image.
            labels = tf.reshape(tf.to_int32(lbl_data), [batch_size, 1])
            return {'imgs': img_data, 'lbl': labels}
        elif mode_str == 'predict' and cfg.dataset == 'cxidb':
            # if predicting and using the cxidb dataset
            # the 'labels' field is is filled with the unique
            # bunchID of the image.
            labels = tf.reshape(
                tf.argmax(lbl_data, axis=-1, output_type=tf.int32),
                [batch_size, 1])
            return {'imgs': img_data, 'lbl': labels}

        labels = tf.reshape(
            tf.to_int32(lbl_data), [batch_size, cfg.num_classes])
        return img_data, labels
    else:
        print('Currently --datasets has to be: \'helium\', \'cxidb\'')
        return None, None


def get_dataset(mode_str, cfg):
    '''
    Gets a dataset tuple with instructions for reading airynet.
    Args:
        cfg: the config class
    Returns:
        Single img and lbl
    '''
    if mode_str == 'predict':
        num_epochs = 1
    else:
        num_epochs = None

    if not cfg.fraction_mode or mode_str == 'eval':
        filenames = glob(
            cfg.data_dir + cfg.dataset + '/*' + mode_str + '.tfrecord')
    else:
        filenames = glob(cfg.data_dir + cfg.dataset + '/*' + mode_str + '*{}'.
                         format(cfg.fraction).replace('.', '_') + '.tfrecord')

    if cfg.dataset == 'cxidb' and mode_str == 'predict':
        filenames = glob(cfg.data_dir + cfg.dataset + '/*train.tfrecord')

    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs)
    print(filenames)
    # Allowing None in the signature so that
    # dataset_factory can use the default.
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    img_string = 'image/'
    if cfg.use_ccf:
        img_string += 'ccf'
    else:
        img_string += 'encoded'

    if cfg.add_noise == 0 and cfg.use_ccf:
        img_string += '_noise_'
    elif cfg.add_noise == 1:
        img_string += '_noise_mean'
    elif cfg.add_noise == 2:
        img_string += '_noise_mean_std'
    elif cfg.add_noise == 3:
        img_string += '_noise_max'
    print(img_string)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            img_string: tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64)
        })
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image_data = tf.decode_raw(features[img_string], tf.uint16)
    if mode_str == 'train' or mode_str == 'eval':
        lbl_data = tf.decode_raw(features['image/class/label'], tf.int16)
        lbl_data.set_shape([cfg.num_classes])
    elif mode_str == 'predict':
        if cfg.dataset == 'helium':
            lbl_data = tf.decode_raw(features['image/class/label'], tf.int32)
            lbl_data.set_shape([1])
        elif cfg.dataset == 'cxidb':
            lbl_data = tf.decode_raw(features['image/class/label'], tf.int16)
            lbl_data.set_shape([cfg.num_classes])

    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)

    image_data_shape = tf.stack([height, width, 1])

    image_data = tf.reshape(image_data, image_data_shape)
    image_data.set_shape([cfg.ori_width, cfg.ori_height, cfg.ori_depth])

    return image_data, lbl_data


def get_cifar_dataset(mode_str, cfg):
    '''
    Gets a dataset tuple with instructions for reading airynet.
    Args:
        cfg: the config class
    Returns:
        A `Dataset` namedtuple.
        Raises:
        ValueError: if `split_name` is not a valid train/test split.
    '''
    filenames = glob(cfg.data_dir + cfg.dataset + '/*' + mode_str + '*')

    # Allowing None in the signature so
    # that dataset_factory can use the default.
    reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
        tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label':
        tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image':
        slim.tfexample_decoder.Image(
            shape=[cfg.ori_width, cfg.ori_height, cfg.ori_depth],
            channels=cfg.ori_depth),
        'label':
        slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      items_to_handlers)

    if cfg.mode == 'train':
        split_size = 50000
    elif cfg.mode == 'test':
        split_size = 10000

    desc = {
        'image': 'The image',
        'label': 'The label',
    }
    return slim.dataset.Dataset(
        data_sources=filenames,
        reader=reader,
        decoder=decoder,
        num_samples=split_size,
        items_to_descriptions=desc,
        num_classes=cfg.num_classes)


def augment_data(img_data):
    # img_data = tf.image.random_contrast(img_data, .25, 1.5)
    # img_data = tf.image.random_brightness(img_data, 1.5)
    img_data = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),
                         img_data)
    img_data = tf.map_fn(lambda img: tf.image.random_flip_up_down(img),
                         img_data)
    return img_data
