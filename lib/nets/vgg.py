#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

from tensorcv.models.layers import *
from tensorcv.models.base import BaseModel


VGG_MEAN = [103.939, 116.779, 123.68]


def resize_tensor_image_with_smallest_side(image, small_size):
    """
    Resize image tensor with smallest side = small_size and
    keep the original aspect ratio.

    Args:
        image (tf.tensor): 4-D Tensor of shape
            [batch, height, width, channels]
            or 3-D Tensor of shape [height, width, channels].
        small_size (int): A 1-D int. The smallest side of resize image.

    Returns:
        Image tensor with the original aspect ratio and
        smallest side = small_size.
        If images was 4-D, a 4-D float Tensor of shape
        [batch, new_height, new_width, channels].
        If images was 3-D, a 3-D float Tensor of shape
        [new_height, new_width, channels].
    """
    im_shape = tf.shape(image)
    shape_dim = image.get_shape()
    if len(shape_dim) <= 3:
        height = tf.cast(im_shape[0], tf.float32)
        width = tf.cast(im_shape[1], tf.float32)
    else:
        height = tf.cast(im_shape[1], tf.float32)
        width = tf.cast(im_shape[2], tf.float32)

    height_smaller_than_width = tf.less_equal(height, width)

    new_shorter_edge = tf.constant(small_size, tf.float32)
    new_height, new_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, (width / height) * new_shorter_edge),
        lambda: ((height / width) * new_shorter_edge, new_shorter_edge))

    return tf.image.resize_images(
        tf.cast(image, tf.float32),
        [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)])


class BaseVGG(BaseModel):
    """ base of VGG class """
    def __init__(self, num_class=1000,
                 num_channels=3,
                 im_height=224, im_width=224,
                 learning_rate=0.0001,
                 is_load=False,
                 pre_train_path=None,
                 is_rescale=False):
        """
        Args:
            num_class (int): number of image classes
            num_channels (int): number of input channels
            im_height, im_width (int): size of input image
                               Can be unknown when testing.
            learning_rate (float): learning rate of training
        """

        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.im_height = im_height
        self.im_width = im_width
        self.num_class = num_class
        self._is_rescale = is_rescale

        self.layer = {}

        self._is_load = is_load
        if self._is_load and pre_train_path is None:
            raise ValueError('pre_train_path can not be None!')
        self._pre_train_path = pre_train_path

        self.set_is_training(True)

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, name='image',
            shape=[None, self.im_height, self.im_width, self.num_channels])

        self.label = tf.placeholder(tf.int64, [None], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob=0.5)
        self.set_train_placeholder([self.image, self.label])
        self.set_prediction_placeholder(self.image)


class VGG19(BaseVGG):

    def _create_conv(self, input_im, data_dict):

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv], nl=tf.nn.relu,
                       trainable=True, data_dict=data_dict):
            conv1_1 = conv(input_im, 3, 64, 'conv1_1')
            conv1_2 = conv(conv1_1, 3, 64, 'conv1_2')
            pool1 = max_pool(conv1_2, 'pool1', padding='SAME')

            conv2_1 = conv(pool1, 3, 128, 'conv2_1')
            conv2_2 = conv(conv2_1, 3, 128, 'conv2_2')
            pool2 = max_pool(conv2_2, 'pool2', padding='SAME')

            conv3_1 = conv(pool2, 3, 256, 'conv3_1')
            conv3_2 = conv(conv3_1, 3, 256, 'conv3_2')
            conv3_3 = conv(conv3_2, 3, 256, 'conv3_3')
            conv3_4 = conv(conv3_3, 3, 256, 'conv3_4')
            pool3 = max_pool(conv3_4, 'pool3', padding='SAME')

            conv4_1 = conv(pool3, 3, 512, 'conv4_1')
            conv4_2 = conv(conv4_1, 3, 512, 'conv4_2')
            conv4_3 = conv(conv4_2, 3, 512, 'conv4_3')
            conv4_4 = conv(conv4_3, 3, 512, 'conv4_4')
            pool4 = max_pool(conv4_4, 'pool4', padding='SAME')

            conv5_1 = conv(pool4, 3, 512, 'conv5_1')
            conv5_2 = conv(conv5_1, 3, 512, 'conv5_2')
            conv5_3 = conv(conv5_2, 3, 512, 'conv5_3')
            conv5_4 = conv(conv5_3, 3, 512, 'conv5_4')
            pool5 = max_pool(conv5_4, 'pool5', padding='SAME')

            self.layer['conv1_2'] = conv1_2
            self.layer['conv2_2'] = conv2_2
            self.layer['conv3_4'] = conv3_4
            self.layer['conv4_4'] = conv4_4
            self.layer['pool5'] = pool5
            self.layer['conv_out'] = self.layer['conv5_4'] = conv5_4

        return pool5

    def _create_model(self):

        with tf.name_scope('input'):
            input_im = self.model_input[0]
            keep_prob = self.model_input[1]

            input_im = tf.reshape(input_im, [-1, 224, 224, 3])

            self.layer['input'] = input_im
            # Convert RGB image to BGR image
            red, green, blue = tf.split(axis=3,
                                        num_or_size_splits=3,
                                        value=input_im)

            input_bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])

        data_dict = {}
        if self._is_load:
            data_dict = np.load(self._pre_train_path,
                                encoding='latin1').item()

        conv_output = self._create_conv(input_bgr, data_dict)

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([fc], trainable=True, data_dict=data_dict):
            fc6 = fc(conv_output, 4096, 'fc6', nl=tf.nn.relu)
            dropout_fc6 = dropout(fc6, keep_prob, self.is_training)

            fc7 = fc(dropout_fc6, 4096, 'fc7', nl=tf.nn.relu)
            dropout_fc7 = dropout(fc7, keep_prob, self.is_training)

            fc8 = fc(dropout_fc7, self.num_class, 'fc8')

            self.layer['fc6'] = fc6
            self.layer['fc7'] = fc7
            self.layer['fc8'] = self.layer['output'] = fc8


class VGG19_FCN(VGG19):

    def _create_model(self):

        with tf.name_scope('input'):
            input_im = self.model_input[0]
            keep_prob = self.model_input[1]

            if self._is_rescale:
                input_im =\
                    resize_tensor_image_with_smallest_side(input_im, 224)
            self.layer['input'] = input_im

            # Convert rgb image to bgr image
            red, green, blue = tf.split(axis=3, num_or_size_splits=3,
                                        value=input_im)

            input_bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])

        data_dict = {}
        if self._is_load:
            data_dict = np.load(self._pre_train_path,
                                encoding='latin1').item()

        conv_outptu = self._create_conv(input_bgr, data_dict)

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv], trainable=True,
                       data_dict=data_dict, padding='VALID'):

            fc6 = conv(conv_outptu, 7, 4096, 'fc6', nl=tf.nn.relu)
            dropout_fc6 = dropout(fc6, keep_prob, self.is_training)

            fc7 = conv(dropout_fc6, 1, 4096, 'fc7', nl=tf.nn.relu)
            dropout_fc7 = dropout(fc7, keep_prob, self.is_training)

            fc8 = conv(dropout_fc7, 1, self.num_class, 'fc8')

            self.layer['fc6'] = fc6
            self.layer['fc7'] = fc7
            self.layer['fc8'] = self.layer['output'] = fc8

        self.output = tf.identity(fc8, 'model_output')

        self.avg_output = global_avg_pool(fc8)
