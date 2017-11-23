#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test_cam.py
# Author: Qian Ge <geqian1001@gmail.com>

# import argparse
from collections import namedtuple
import tensorflow as tf

from tensorcv.dataflow.image import ImageLabelFromFolder
from tensorcv.callbacks import *
from tensorcv.train.config import TrainConfig
from tensorcv.train.simple import SimpleFeedTrainer
from tensorcv.predicts.config import PridectConfig
from tensorcv.predicts.simple import SimpleFeedPredictor
from tensorcv.predicts import *

from setup_test_env import *
from models.cam import VGGCAM

configpath = namedtuple('CONFIG_PATH', ['summary_dir'])
config_path = configpath(summary_dir=SAVE_DIR)

NUM_CHANNEL = 3


def get_config(FLAGS):
    # data for training
    dataset_train = ImageLabelFromFolder(FLAGS.type,
                                         data_dir=CLASS_IMPATH,
                                         num_class=FLAGS.nclass,
                                         resize=224,
                                         num_channel=NUM_CHANNEL)

    # Print image class name and label
    # print(dataset_train.label_dict)

    training_callbacks = [
        # TrainSummary(key='train', periodic=1),
        CheckScalar(['accuracy/result', 'loss/result'], periodic=1)]

    inspect_class = None

    return TrainConfig(
        dataflow=dataset_train,
        model=VGGCAM(num_class=FLAGS.nclass,
                     inspect_class=inspect_class,
                     learning_rate=0.001,
                     is_load=False),
        monitors=TFSummaryWriter(),
        callbacks=training_callbacks,
        batch_size=FLAGS.bsize,
        max_epoch=1,
        # summary_periodic=1,
        default_dirs=config_path)


# def get_predict_config(FLAGS):
#     dataset_test = ImageFromFile(FLAGS.type,
#                                  data_dir=config_path.test_data_dir,
#                                  shuffle=False,
#                                  resize=224,
#                                  num_channel=NUM_CHANNEL)
#     # dataset_test = ImageLabelFromFolder('.jpg',
#     #                     data_dir = CLASS_IMPATH,
#     #                     num_class = FLAGS.nclass,
#     #                     resize = 224,
#     #                     num_channel = NUM_CHANNEL)
#     prediction_list = [
#         # PredictionScalar(['pre_label'], ['label']),
#         # PredictionMeanScalar('accuracy/result', 'test_accuracy'),
#         PredictionMat('classmap/result', ['test']),
#         PredictionOverlay(['classmap/result', 'image'], ['map', 'image'],
#                           color=True, merge_im=True),
#         PredictionImage(['image'], ['image'], color=True, merge_im=True)]

#     return PridectConfig(
#         dataflow=dataset_test,
#         model=VGGCAM(num_class=FLAGS.nclass, inspect_class=FLAGS.label,
#                      is_load=True, pre_train_path=config_path.vgg_dir),
#         model_name=FLAGS.model,
#         predictions=prediction_list,
#         batch_size=FLAGS.bsize,
#         default_dirs=config_path)


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--bsize', default=1, type=int)
#     parser.add_argument('--label', default=-1, type=int,
#                         help='Label of inspect class.')
#     parser.add_argument('--nclass', default=1, type=int,
#                         help='number of image class')

#     parser.add_argument('--type', default='.jpg', type=str,
#                         help='image type for training and testing')

#     parser.add_argument('--model', type=str,
#                         help='file name of the trained model')

#     return parser.parse_args()


def test_cam():
    inargs = namedtuple('IN_ARGS', ['bsize', 'label', 'nclass', 'type'])
    FLAGS = inargs(bsize=1, label=-1, nclass=1, type='.jpg')

    # FLAGS = get_args()
    config = get_config(FLAGS)
    SimpleFeedTrainer(config).train()
    tf.reset_default_graph()
    #
    # if FLAGS.train:
    #     config = get_config(FLAGS)
    #     SimpleFeedTrainer(config).train()
    # if FLAGS.predict:
    #     config = get_predict_config(FLAGS)
    #     SimpleFeedPredictor(config).run_predict()

# 0.6861924529075623
