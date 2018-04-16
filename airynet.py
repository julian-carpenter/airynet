'''
Runs a airynet model on the CIFAR-10 or the helium dataset.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime

import nn
import tensorflow as tf
from tqdm import tqdm
from numpy import save
from scipy.io import savemat


def get_time():
    return datetime.now().strftime('%m%d_%H%M%S')


def main(cfg):
    # Using the Winograd non-fused algorithms
    # provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    if os.path.isdir(cfg.out_dir_name):
        if cfg.out_dir_name == cfg.load_dir:
            model_dir = cfg.load_dir
        else:
            print(
                'Overriding output path {} with: {} since it already existed '
                'and was not the loading path'.format(
                    cfg.out_dir_name, cfg.out_dir_name + '_tmp'))
            cfg.out_dir_name = cfg.out_dir_name + '_tmp'

    if cfg.mode == 'train' and not cfg.load_dir:
        if cfg.out_dir_name:
            model_dir = '{}/{}_{}'.format(cfg.log_dir, cfg.dataset,
                                          cfg.out_dir_name)
        else:
            model_dir = '{}/{}_{}'.format(cfg.log_dir, cfg.dataset, get_time())
    else:
        model_dir = cfg.load_dir

    print(model_dir)

    with tf.device('/cpu:0'):
        run_config_obj = tf.contrib.learn.RunConfig(
            session_config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False),
            save_checkpoints_secs=3600,
            model_dir=model_dir,
            tf_random_seed=cfg.random_seed,
            keep_checkpoint_every_n_hours=2,
            keep_checkpoint_max=10)
        airynet_classifier = tf.estimator.Estimator(
            config=run_config_obj,
            model_fn=nn.model_fn,
            model_dir=model_dir,
            params={'config': cfg})

        if cfg.mode == 'train':
            for _ in range(int(cfg.train_steps) // int(cfg.steps_per_eval)):
                tensors_to_log = {
                    'learning_rate': 'learning_rate',
                    'cross_entropy': 'cross_entropy',
                    'train_accuracy': 'train_accuracy',
                    'confusion_matrix': 'confusion_matr'
                }
                print(cfg.l1_regularization_scale, cfg.l2_regularization_scale,
                      cfg.resnet_size, cfg.alpha, cfg.relu_leakiness,
                      cfg.use_ccf, cfg.add_noise)
                logging_hook = tf.train.LoggingTensorHook(
                    tensors=tensors_to_log, every_n_iter=cfg.log_step)

                airynet_classifier.train(
                    input_fn=lambda: nn.get_queue('train', cfg),
                    steps=cfg.steps_per_eval,
                    hooks=[logging_hook])

                # Evaluate the model and print results
                eval_results = airynet_classifier.evaluate(
                    input_fn=lambda: nn.get_queue('eval', cfg), steps=100)
                print(eval_results)
        elif cfg.mode == 'predict':
            # Use the trained network to predict
            predict_results = airynet_classifier.predict(
                input_fn=lambda: nn.get_queue('predict', cfg),
                hooks=[nn.PredictionHook()])
            with open(
                    os.path.join(os.path.curdir, 'predictions',
                                 '{}.txt'.format(cfg.dataset)),
                    'w') as text_file:
                print('Bunch_ID:\tClasses:\tProbabilities:', file=text_file)
                out_dict = {}
                for idx, shot in tqdm(enumerate(predict_results)):
                    out_dict.update({
                        'shot_{:05d}'.format(idx + 1): [
                            shot['bunchID'], shot['classes'],
                            shot['probabilities']
                        ]
                    })
                    out_str = ('{}\t{}\t{}'.format(
                        shot['bunchID'],
                        ['{}'.format(int(k)) for k in shot['classes']],
                        ['{:.04f}'.format(k)
                         for k in shot['probabilities']])).replace('\'', '')
                    print(
                        out_str.replace('[', '').replace(']', ''),
                        file=text_file)
                save(
                    os.path.join(os.path.curdir, 'predictions',
                                 '{}.npy'.format(cfg.dataset)), out_dict)
            #     savemat(
            #         os.path.join(os.path.curdir, 'predictions',
            #                      '{}.mat'.format(cfg.dataset)), out_dict)
        elif cfg.mode == 'save':
            # Use the trained network to predict
            # path_to_saved_model = airynet_classifier.export_savedmodel(
            #     export_dir_base=cfg.save_dir,
            #     serving_input_receiver_fn=nn.get_queue(
            #         'serve'))  # not functional yet
            # print(path_to_saved_model)
            print('Not yet implemented')


if __name__ == '__main__':

    cfg, _ = nn.get_config()
    if cfg.log_level == 'WARN':
        verbosity = tf.logging.WARN
    elif cfg.log_level == 'DEBUG':
        verbosity = tf.logging.DEBUG
    else:
        verbosity = tf.logging.INFO

    tf.logging.set_verbosity(verbosity)
    main(cfg)
