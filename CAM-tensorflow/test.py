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

import model
import config

def get_config(FLAGS):
    dataset_train = MNISTLabel('train', config.data_dir)
    dataset_val = MNISTLabel('val', config.data_dir)
    dataset_test = ImageFromFile('.png', 
                                data_dir = config.data_dir, 
                                shuffle = False,
                                normalize_fnc = normalize_one,
                                num_channel = 1)

    inference_list = [InferScalars('accuracy/result', 'test_accuracy')]
    infer_list = InferImages('classmap/result','image', color = True)
    return TrainConfig(
                 dataflow = dataset_train, 
                 model = model.mnistCAM(learning_rate = 0.001, inspect_class = FLAGS.label),
                 monitors = TFSummaryWriter(),
                 callbacks = [
                    ModelSaver(periodic = 100),
                    TrainSummary(key = 'train', periodic = 10),
                    FeedInferenceBatch(dataset_val, periodic = 100, batch_count = 100, 
                                  # extra_cbs = TrainSummary(key = 'test'),
                                  inferencers = inference_list),
                    FeedInference(dataset_test, periodic = 100,
                                  infer_batch_size = 1, 
                                  inferencers = infer_list),
                    CheckScalar(['accuracy/result','loss/result'], periodic = 100),
                  ],
                 batch_size = FLAGS.batch_size, 
                 max_epoch = 100,
                 summary_periodic = 100,
                 default_dirs = config)

def get_predict_config(FLAGS):
    # dataset_test = MNISTLabel('test', config.data_dir, shuffle = False)
    dataset_test = ImageFromFile('.png', 
                                data_dir = config.data_dir, 
                                shuffle = False,
                                normalize_fnc = normalize_one,
                                num_channel = 1)
    prediction_list = [
             # PredictionScalar(['pre_label'], ['label']),
             # PredictionMeanScalar('accuracy/result', 'test_accuracy'),
             PredictionMat('classmap/result', ['test']),
             PredictionImage(['classmap/result', 'image'], ['map', 'image'], merge_im = True)
             ]

    return PridectConfig(
                dataflow = dataset_test,
                model = model.mnistCAM(inspect_class = FLAGS.label),
                model_name = 'model-6000',
                predictions = prediction_list,
                batch_size = FLAGS.batch_size,
                default_dirs = config)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--label', default = 1, type = int,
                        help = 'Label of inspect class.')

    parser.add_argument('--predict', action = 'store_true', 
                        help = 'Run prediction')
    parser.add_argument('--train', action = 'store_true', 
                        help = 'Train the model')

    return parser.parse_args()

if __name__ == '__main__':
    FLAGS = get_args()

    if FLAGS.train:
        config = get_config(FLAGS)
        SimpleFeedTrainer(config).train()
    elif FLAGS.predict:
        config = get_predict_config(FLAGS)
        SimpleFeedPredictor(config).run_predict()

    
