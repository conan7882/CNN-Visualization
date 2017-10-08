# File: vgg.py
# Author: Qian Ge <geqian1001@gmail.com>
import argparse

import tensorflow as tf

import tensorcv
from tensorcv.dataflow.image import *
from tensorcv.callbacks import *
from tensorcv.train.config import TrainConfig
from tensorcv.train.simple import SimpleFeedTrainer
from tensorcv.predicts.simple import SimpleFeedPredictor
from tensorcv.predicts import *

from model import VGGCAM
import configvggtitan as config_path

NUM_CLASS = 257
NUM_CHANNEL = 3

def get_config(FLAGS):
    dataset_train = ImageLabelFromFolder('.jpg', data_dir = config_path.data_dir, 
                        num_class = NUM_CLASS,
                        reshape = 224,
                        num_channel = NUM_CHANNEL)

    dataset_val = ImageLabelFromFolder('.jpg', data_dir = config_path.data_dir, 
                        num_class = NUM_CLASS,
                        reshape = 224,
                        num_channel = NUM_CHANNEL)
    dataset_test = ImageFromFile('.jpg', 
                                data_dir = config_path.test_data_dir, 
                                shuffle = False,
                                num_channel = NUM_CHANNEL)

    inference_list_validation = [InferScalars('accuracy/result', 'test_accuracy')]
    inference_list_test = InferImages('classmap/result','image', color = True)

    return TrainConfig(
                 dataflow = dataset_train, 
                 model = VGGCAM(num_class = NUM_CLASS, 
                           inspect_class = FLAGS.label,
                           learning_rate = 0.001,
                           is_load = True,
                           pre_train_path = config_path.vgg_dir),
                 monitors = TFSummaryWriter(),
                 callbacks = [
                    ModelSaver(periodic = 100),
                    TrainSummary(key = 'train', periodic = 50),
                    FeedInferenceBatch(dataset_val, 
                                  periodic = 100, 
                                  batch_count = 10, 
                                  # extra_cbs = TrainSummary(key = 'test'),
                                  inferencers = inference_list_validation),
                    FeedInference(dataset_test, periodic = 50,
                                  infer_batch_size = 1, 
                                  inferencers = inference_list_test),
                    CheckScalar(['accuracy/result','loss/result'], 
                                 periodic = 10),
                  ],
                 batch_size = FLAGS.bsize, 
                 max_epoch = 100,
                 summary_periodic = 50,
                 default_dirs = config_path)

def get_predict_config(FLAGS):
    dataset_test = ImageFromFile('.jpg', 
                                data_dir = config_path.test_data_dir, 
                                shuffle = False,
                                num_channel = 3)
    # dataset_test = ImageLabelFromFolder('.jpg', data_dir = config_path.data_dir, 
    #                     num_class = NUM_CLASS,
    #                     reshape = 224,
    #                     num_channel = NUM_CHANNEL)
    prediction_list = [
             # PredictionScalar(['pre_label'], ['label']),
             # PredictionMeanScalar('accuracy/result', 'test_accuracy'),
             PredictionMat('classmap/result', ['test']),
             PredictionOverlay(['classmap/result', 'image'], ['map', 'image'], color = True, merge_im = False)
             ]

    return PridectConfig(
                dataflow = dataset_test,
                model = VGGCAM(num_class = NUM_CLASS, 
                           inspect_class = FLAGS.label,
                           learning_rate = 0.001,
                           is_load = True,
                           pre_train_path = config_path.vgg_dir),
                model_name = 'model-8400',
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

    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.train:
        config = get_config(FLAGS)
        SimpleFeedTrainer(config).train()
    if FLAGS.predict:
        config = get_predict_config(FLAGS)
        SimpleFeedPredictor(config).run_predict()

# 0.6861924529075623

