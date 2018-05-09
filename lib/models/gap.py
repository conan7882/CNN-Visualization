#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gap.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.models.base import BaseModel
from tensorcv.models.layers import new_weights, global_avg_pool, conv, dropout, max_pool
from tensorcv.models.layers import batch_norm

from lib.nets.vgg import BaseVGG19
from lib.nets.googlenet import BaseGoogLeNet


def mlpconv(inputs, filter_size, hidden_size, wd=0, name='mlpconv'):
    if not isinstance(hidden_size, list):
        hidden_size = [hidden_size]
    with tf.variable_scope(name):
        l_out = conv(inputs,
                     filter_size,
                     hidden_size[0],
                     'microlayer_0',
                     nl=tf.nn.relu,
                     wd=wd)
        for layer_id in range(1, len(hidden_size)):
            l_out = conv(l_out,
                         1,
                         hidden_size[layer_id],
                         'microlayer_{}'.format(layer_id),
                         nl=tf.nn.relu,
                         wd=wd)

        return l_out


class GAPNet(BaseModel):
    def __init__(self, num_class=10, wd=0):
        self._n_class = num_class
        self._wd = wd
        # self._pre_train_path = pre_train_path

        self.set_is_training(True)
        self.layer = {}

    def set_is_training(self, is_training):
        self._is_traing = is_training

    def create_model(self, input_dict):
        self._input_dict = input_dict
        self._create_model()

    def _create_conv(self, inputs):
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        mlpconv_1 = mlpconv(
            inputs,
            filter_size=8,
            hidden_size=[96, 96],
            name='mlpconv_1',
            wd=self._wd)
        # mlpconv_1 = mlpconv(
        #     inputs,
        #     filter_size=5,
        #     hidden_size=[192, 160, 96],
        #     name='mlpconv_1',
        #     wd=self._wd)
        mlpconv_1 = max_pool(mlpconv_1, 'pool1', padding='SAME')
        mlpconv_1 = dropout(mlpconv_1, self.dropout, self._is_traing)
        mlpconv_1 = batch_norm(mlpconv_1, train=self._is_traing, name='bn_1')

        mlpconv_2 = mlpconv(
            mlpconv_1,
            filter_size=8,
            hidden_size=[192, 192],
            name='mlpconv_2',
            wd=self._wd)
        # mlpconv_2 = mlpconv(
        #     mlpconv_1,
        #     filter_size=5,
        #     hidden_size=[192, 192, 192],
        #     name='mlpconv_2',
        #     wd=self._wd)
        mlpconv_2 = max_pool(mlpconv_2, 'pool2', padding='SAME')
        mlpconv_2 = dropout(mlpconv_2, self.dropout, self._is_traing)
        mlpconv_2 = batch_norm(mlpconv_2, train=self._is_traing, name='bn_2')

        mlpconv_3 = mlpconv(
            mlpconv_2,
            filter_size=5,
            hidden_size=[192, self._n_class],
            name='mlpconv_3',
            wd=self._wd)
        # mlpconv_3 = mlpconv(
        #     mlpconv_2,
        #     filter_size=3,
        #     hidden_size=[192, 192, self._n_class],
        #     name='mlpconv_3',
        #     wd=self._wd)
        # mlpconv_3 = max_pool(mlpconv_3, 'pool3', padding='SAME')
        # mlpconv_3 = dropout(pool3, 0.5, self._is_traing)

        return mlpconv_3

    def _create_model(self):
        inputs = self._input_dict['input']
        conv_out = self._create_conv(inputs)

        # init_b = tf.truncated_normal_initializer(stddev=0.01)
        # conv_gap = conv(conv_out, 3, self._n_class, 'conv_gap',
        #                 nl=tf.nn.relu, wd=0, init_b=init_b)
        gap = global_avg_pool(conv_out)

        self.layer['logits'] = gap
        self.layer['feature'] = conv_out
        self.layer['pred'] = tf.argmax(gap, name='pred', axis=-1)
        self.layer['prob'] = tf.nn.softmax(gap, name='prob')

    def _get_loss(self):
        label = self._input_dict['label']
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits= self.layer['logits'], labels=label)
            cross_entropy_loss = tf.reduce_mean(
                cross_entropy, name='cross_entropy_loss')
            tf.add_to_collection('losses', cross_entropy_loss)
            return tf.add_n(tf.get_collection('losses'), name='result')

    def get_loss(self):
        try:
            return self.loss
        except AttributeError:
            self.loss = self._get_loss()
            return self.loss

    def get_train_op(self):
        self.lr = tf.placeholder(tf.float32, name='lr')
        # opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        opt = tf.train.AdamOptimizer(
            beta1=0.5, learning_rate=self.lr)
        loss = self.get_loss()
        return opt.minimize(loss)

    def get_accuracy(self):
        label = self._input_dict['label']
        pred = self.layer['pred']

        correct = tf.equal(label, pred)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy


