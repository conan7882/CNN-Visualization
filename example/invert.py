#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: invert.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import scipy
import numpy as np
import tensorflow as tf

import config_path as config

import sys
sys.path.append('../')
import lib.utils.viz as viz
import lib.utils.normalize as normlize
from lib.nets.googlenet import BaseGoogLeNet
from lib.models.invert import InvertCNN
import lib.utils.viz as viz
import lib.utils.normalize as normlize


file_path = os.path.join(config.im_path, 'im_0.png')
MEAN = [103.939, 116.779, 123.68]

if __name__ == '__main__':
    im = [scipy.misc.imread(file_path)]
    input_mean = np.mean(im)
    input_std = np.std(im)
    layer_key = 'inception5b'
    cnn_model = BaseGoogLeNet(config.googlenet_path)
    invert_model = InvertCNN(
        224, 224, 3,
        input_mean=input_mean,
        input_std=input_std,
        mean_list=MEAN)

    input_im = tf.placeholder(tf.float32, [1, 224, 224, 3], name='input')
    

    feat_im = cnn_model.get_feature_map(input_im, layer_key)
    feat_invert = cnn_model.get_feature_map(invert_model.invert_im, layer_key)
    
    train_op = invert_model.optimize_image(feat_invert, feat_im)
    result_op = invert_model.get_opt_im()

    writer = tf.summary.FileWriter(config.save_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # writer.add_graph(sess.graph)

        for step in range(0, 1000):
            
            _, loss, loss1, loss2 = sess.run(
                [train_op,
                 invert_model.loss,
                 invert_model.mse_loss,
                 invert_model.vt_loss],
                feed_dict={input_im:im})
            print(step, loss, loss1, loss2)
            opt_im = sess.run(result_op)
            if step % 10 == 0:
                # opt_im = np.clip(opt_im, 0, 255)
                # 
                # opt_im = opt_im * input_std + input_mean
                # print(opt_im)
                scipy.misc.imsave(os.path.join(config.save_path, 'test_{}.png'.format(step)),
                                  np.squeeze(opt_im))
        
