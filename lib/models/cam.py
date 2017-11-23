#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cam.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import numpy as np

from tensorcv.models.layers import new_weights, dropout, global_avg_pool, conv, max_pool
from tensorcv.models.base import BaseModel


class BaseCAM(BaseModel):
    """ base of class activation map class """
    def __init__(self, num_class=10,
                 inspect_class=None,
                 num_channels=1,
                 learning_rate=0.0001):

        self._learning_rate = learning_rate
        self._num_channels = num_channels
        self._num_class = num_class
        self._inspect_class = inspect_class

        self.set_is_training(True)

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, name='image',
            shape=[None, None, None, self._num_channels])
        self.label = tf.placeholder(tf.int64, [None], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob=0.5)
        self.set_train_placeholder([self.image, self.label])
        self.set_prediction_placeholder([self.image, self.label])

    def _create_conv(self, input_im):
        raise NotImplementedError()

    def _get_loss(self):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.output, labels=self.label)
            cross_entropy_loss = tf.reduce_mean(
                cross_entropy, name='cross_entropy_loss')
            tf.add_to_collection('losses', cross_entropy_loss)
            return tf.add_n(tf.get_collection('losses'), name='result')

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(
            beta1=0.5, learning_rate=self._learning_rate)

    def _ex_setup_graph(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.prediction, self.label)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='result')

    def _setup_summary(self):
        tf.summary.scalar("train_accuracy", self.accuracy,
                          collections=['train'])

    def get_classmap(self, label, conv_out, input_im):
        """
        Compute class activation map of class = label with name 'classmap'

        Args:
            label (int): a scalar int indicate the class label
            conv_out (tf.tensor): 4-D Tensor of shape
                [batch, height, width, channels]. Output of
                convolutional layers.
            input_im (tf.tensor): A 4-D Tensor image.
                The original model input image patch.
        """
        # Get original image size used for interpolation
        o_height = tf.shape(input_im)[1]
        o_width = tf.shape(input_im)[2]

        # Get shape of output of convolution layers
        conv_out_channel = tf.shape(conv_out)[-1]
        conv_height = tf.shape(conv_out)[1]
        conv_width = tf.shape(conv_out)[2]

        # Get weights corresponding to class = label
        with tf.variable_scope('cam') as scope:
            scope.reuse_variables()
            label_w = tf.gather(
                tf.transpose(tf.get_variable('weights')), label)
            label_w = tf.reshape(label_w, [-1, conv_out_channel, 1])
            label_w = tf.tile(label_w, [tf.shape(conv_out)[0], 1, 1])

        conv_reshape = tf.reshape(
            conv_out, [-1, conv_height * conv_width, conv_out_channel])
        classmap = tf.matmul(conv_reshape, label_w)

        # Interpolate to orginal size
        classmap = tf.reshape(classmap, [-1, conv_height, conv_width, 1])
        classmap = tf.image.resize_bilinear(classmap,
                                            [o_height, o_width],
                                            name='result')

class VGGCAM(BaseCAM):
    def __init__(self, num_class=1000,
                 inspect_class=None,
                 num_channels=3,
                 learning_rate=0.0001,
                 is_load=True,
                 pre_train_path=None):

        self._is_load = is_load
        if self._is_load and pre_train_path is None:
            raise ValueError('pre_train_path can not be None!')
        self._pre_train_path = pre_train_path

        super(VGGCAM, self).__init__(num_class=num_class,
                                     inspect_class=inspect_class,
                                     num_channels=num_channels,
                                     learning_rate=learning_rate)

    def _create_conv(self, input_im):

        VGG_MEAN = [103.939, 116.779, 123.68]

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

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv], nl=tf.nn.relu,
                       trainable=False, data_dict=data_dict):
            conv1_1 = conv(input_bgr, 3, 64, 'conv1_1')
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

        return conv5_4

    def _create_model(self):

        input_im = self.model_input[0]
        keep_prob = self.model_input[1]

        conv_out = self._create_conv(input_im)

        init_b = tf.truncated_normal_initializer(stddev=0.01)
        conv_cam = conv(conv_out, 3, 1024, 'conv_cam',
                        nl=tf.nn.relu, wd=0.01, init_b=init_b)
        gap = global_avg_pool(conv_cam)
        dropout_gap = dropout(gap, keep_prob, self.is_training)

        with tf.variable_scope('cam'):
            init = tf.truncated_normal_initializer(stddev=0.01)
            fc_w = new_weights(
                'weights', 1,
                [gap.get_shape().as_list()[-1], self._num_class],
                initializer=init, wd=0.01)
            fc_cam = tf.matmul(dropout_gap, fc_w, name='output')

        self.output = tf.identity(fc_cam, 'model_output')
        self.prediction = tf.argmax(fc_cam, name='pre_label', axis=-1)
        self.prediction_pro = tf.nn.softmax(fc_cam, name='pre_pro')

        if self._inspect_class is not None:
            with tf.name_scope('classmap'):
                self.get_classmap(self._inspect_class, conv_cam, input_im)


# if __name__ == '__main__':
#     num_class = 257
#     num_channels = 3

#     vgg_cam_model = VGGCAM(num_class=num_class,
#                            inspect_class=None,
#                            num_channels=num_channels,
#                            learning_rate=0.0001,
#                            is_load=True,
#                            pre_train_path='E:\\GITHUB\\workspace\\CNN\pretrained\\vgg19.npy')

#     vgg_cam_model.create_graph()

#     grads = vgg_cam_model.get_grads()
#     opt = vgg_cam_model.get_optimizer()
#     train_op = opt.apply_gradients(grads, name='train')

#     writer = tf.summary.FileWriter('E:\\GITHUB\\workspace\\CNN\\other\\')
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         writer.add_graph(sess.graph)
#     writer.close()
