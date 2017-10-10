# File: test.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse

import numpy as np
import tensorflow as tf

from tensorcv.dataflow import *
from tensorcv.train.simple import SimpleFeedTrainer
from tensorcv.train.config import TrainConfig
from tensorcv.predicts import *
from tensorcv.predicts.simple import SimpleFeedPredictor
from tensorcv.callbacks import *

import CAM
import configmnist as config_path

def get_config(FLAGS):
    dataset_train = MNISTLabel('train', config_path.data_dir)
    dataset_val = MNISTLabel('val', config_path.data_dir)
    dataset_test = ImageFromFile('.png', 
                                data_dir = config_path.infer_data_dir, 
                                shuffle = False,
                                normalize_fnc = normalize_one,
                                num_channel = 1)

    inference_list = [InferScalars('accuracy/result', 'test_accuracy')]
    # infer_list = InferImages('classmap/result','image', color = True)
    inference_list_test = [
           InferOverlay(['classmap/result', 'image'], ['map', 'image'], color = True),
           InferImages('classmap/result', 'map', color = True)
        ]
    return TrainConfig(
                 dataflow = dataset_train, 
                 model = CAM.mnistCAM(learning_rate = 0.0001, inspect_class = FLAGS.label),
                 monitors = TFSummaryWriter(),
                 callbacks = [
                    ModelSaver(periodic = 100),
                    TrainSummary(key = 'train', periodic = 10),
                    FeedInferenceBatch(dataset_val, periodic = 100, batch_count = 100, 
                                  # extra_cbs = TrainSummary(key = 'test'),
                                  inferencers = inference_list),
                    FeedInference(dataset_test, periodic = 100,
                                  infer_batch_size = 4, 
                                  inferencers = inference_list_test),
                    CheckScalar(['accuracy/result','loss/result'], periodic = 100),
                  ],
                 batch_size = FLAGS.bsize, 
                 max_epoch = 25,
                 summary_periodic = 100,
                 default_dirs = config_path)

def get_predict_config(FLAGS):
    # dataset_test = MNISTLabel('test', config.data_dir, shuffle = False)
    dataset_test = ImageFromFile('.png', 
                                data_dir = config_path.test_data_dir, 
                                shuffle = False,
                                normalize_fnc = normalize_one,
                                num_channel = 1)
    prediction_list = [
             # PredictionScalar(['pre_label'], ['label']),
             # PredictionMeanScalar('accuracy/result', 'test_accuracy'),
             PredictionMat('classmap/result', ['test']),
             PredictionOverlay(['classmap/result', 'image'], ['map', 'image'], color = True, merge_im = True)
             ]

    return PridectConfig(
                dataflow = dataset_test,
                model = CAM.mnistCAM(inspect_class = FLAGS.label),
                model_name = FLAGS.model,
                predictions = prediction_list,
                batch_size = FLAGS.bsize,
                default_dirs = config_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bsize', default = 32, type = int)
    parser.add_argument('--label', default = 1, type = int,
                        help = 'Label of inspect class.')

    parser.add_argument('--predict', action = 'store_true', 
                        help = 'Run prediction')
    parser.add_argument('--train', action = 'store_true', 
                        help = 'Train the model')

    parser.add_argument('--model', type = str, 
                        help = 'file name of the trained model')

    return parser.parse_args()

if __name__ == '__main__':
    FLAGS = get_args()

    if FLAGS.train:
        config = get_config(FLAGS)
        SimpleFeedTrainer(config).train()
    elif FLAGS.predict:
        config = get_predict_config(FLAGS)
        SimpleFeedPredictor(config).run_predict()

    
