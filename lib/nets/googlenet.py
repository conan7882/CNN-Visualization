#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: googlenet.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tensorcv.models.layers import conv, fc, global_avg_pool, dropout, max_pool
from tensorcv.models.base import BaseModel


MEAN = [103.939, 116.779, 123.68]

@add_arg_scope
def inception_layer(inputs,
                    conv_11_size,
                    conv_33_reduce_size, conv_33_size,
                    conv_55_reduce_size, conv_55_size,
                    pool_size,
                    data_dict={},
                    trainable=False,
                    name='inception'):

    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([conv], nl=tf.nn.relu, trainable=trainable,
                   data_dict=data_dict):
        conv_11 = conv(inputs, 1, conv_11_size, '{}_1x1'.format(name))

        conv_33_reduce = conv(inputs, 1, conv_33_reduce_size,
                              '{}_3x3_reduce'.format(name))
        conv_33 = conv(conv_33_reduce, 3, conv_33_size, '{}_3x3'.format(name))

        conv_55_reduce = conv(inputs, 1, conv_55_reduce_size,
                              '{}_5x5_reduce'.format(name))
        conv_55 = conv(conv_55_reduce, 5, conv_55_size, '{}_5x5'.format(name))

        pool = max_pool(inputs, '{}_pool'.format(name), stride=1,
                        padding='SAME', filter_size=3)
        convpool = conv(pool, 1, pool_size, '{}_pool_proj'.format(name))

    return tf.concat([conv_11, conv_33, conv_55, convpool],
                     3, name='{}_concat'.format(name))

class BaseGoogLeNet(BaseModel):
    def __init__(self, pre_train_path, is_load=True):
        self.data_dict = {}
        if is_load:
            assert pre_train_path is not None
            self.data_dict = np.load(pre_train_path,
                                encoding='latin1').item()

        self.inputs = tf.placeholder(tf.float32,
                                     [None, None, None, 3],
                                     name='input')

        
        input_bgr = self._sub_mean(self.inputs)
        self._creat_googlenet(input_bgr, self.data_dict)

    def _sub_mean(self, inputs):
        with tf.name_scope('input'):
            input_im = inputs

            # Convert RGB image to BGR image
            red, green, blue = tf.split(axis=3,
                                        num_or_size_splits=3,
                                        value=input_im)

            input_bgr = tf.concat(axis=3, values=[
                blue - MEAN[0],
                green - MEAN[1],
                red - MEAN[2],
            ])
            return input_bgr

    def get_feature_map(self, inputs, layer_key):
        assert layer_key in self.conv_layer
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            # print(tf.get_default_graph().get_name_scope())
            scope.reuse_variables()
            inputs = self._sub_mean(inputs)
            self._creat_googlenet(inputs, self.data_dict)
            return self.conv_layer[layer_key]

    def _creat_googlenet(self,
                         inputs,
                         data_dict,
                         trainable=False):
        self.conv_layer = {}

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv], trainable=trainable,
                       data_dict=data_dict, nl=tf.nn.relu):
            conv1 = conv(inputs, 7, 64, name='conv1_7x7_s2', stride=2)
            padding1 = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
            conv1_pad = tf.pad(conv1, padding1, 'CONSTANT')
            pool1 = max_pool(
                conv1_pad, 'pool1', padding='VALID', filter_size=3, stride=2)
            pool1_lrn = tf.nn.local_response_normalization(
                pool1, depth_radius=2, alpha=2e-05, beta=0.75,
                name='pool1_lrn')

            conv2_reduce = conv(pool1_lrn, 1, 64, name='conv2_3x3_reduce')
            conv2 = conv(conv2_reduce, 3, 192, name='conv2_3x3')
            padding2 = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
            conv2_pad = tf.pad(conv2, padding1, 'CONSTANT')
            pool2 = max_pool(
                conv2_pad, 'pool2', padding='VALID', filter_size=3, stride=2)
            pool2_lrn = tf.nn.local_response_normalization(
                pool2, depth_radius=2, alpha=2e-05, beta=0.75,
                name='pool2_lrn')

        with arg_scope([inception_layer],
                       trainable=trainable,
                       data_dict=data_dict):
            inception3a = inception_layer(
                pool2_lrn, 64, 96, 128, 16, 32, 32, name='inception_3a')
            inception3b = inception_layer(
                inception3a, 128, 128, 192, 32, 96, 64, name='inception_3b')
            pool3 = max_pool(
                inception3b, 'pool3', padding='SAME', filter_size=3, stride=2)

            inception4a = inception_layer(
                pool3, 192, 96, 208, 16, 48, 64, name='inception_4a')
            inception4b = inception_layer(
                inception4a, 160, 112, 224, 24, 64, 64, name='inception_4b')
            inception4c = inception_layer(
                inception4b, 128, 128, 256, 24, 64, 64, name='inception_4c')
            inception4d = inception_layer(
                inception4c, 112, 144, 288, 32, 64, 64, name='inception_4d')
            inception4e = inception_layer(
                inception4d, 256, 160, 320, 32, 128, 128, name='inception_4e')
            pool4 = max_pool(
                inception4e, 'pool4', padding='SAME', filter_size=3, stride=2)

            inception5a = inception_layer(
                pool4, 256, 160, 320, 32, 128, 128, name='inception_5a')
            inception5b = inception_layer(
                inception5a, 384, 192, 384, 48, 128, 128, name='inception_5b')

            self.conv_layer['conv1_7x7_s2'] = conv1
            self.conv_layer['conv2_3x3'] = conv2
            self.conv_layer['inception3a'] = inception3a
            self.conv_layer['inception3b'] = inception3b
            self.conv_layer['inception4a'] = inception4a
            self.conv_layer['inception4b'] = inception4b
            self.conv_layer['inception4c'] = inception4c
            self.conv_layer['inception4d'] = inception4d
            self.conv_layer['inception4e'] = inception4e
            self.conv_layer['inception5a'] = inception5a
            self.conv_layer['inception5b'] = inception5b

        return inception5b

