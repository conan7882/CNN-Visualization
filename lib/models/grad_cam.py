#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: grad_cam.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.models.layers import global_avg_pool


class BaseGradCAM(object):
    def __init__(self, vis_model=None, num_channel=3):
        self._vis_model = vis_model
        self._nchannel = num_channel

    def create_model(self, inputs):
        self._create_model(inputs)

    def _create_model(self, inputs):
        pass

    def setup_graph(self):
        pass

    def _comp_feature_importance_weight(self, class_id):
        if not isinstance(class_id, list):
            class_id = [class_id]

        with tf.name_scope('feature_weight'):
            self._feature_w_list = []
            for idx, cid in enumerate(class_id):
                one_hot = tf.sparse_to_dense(
                    [[cid, 0]], [self._nclass, 1], 1.0)
                out_act = tf.reshape(self._out_act, [1, self._nclass])
                class_act = tf.matmul(out_act, one_hot,
                                      name='class_act_{}'.format(idx))
                feature_grad = tf.gradients(class_act, self._conv_out,
                                            name='grad_{}'.format(idx))
                feature_grad = tf.squeeze(
                    tf.convert_to_tensor(feature_grad), axis=0)
                feature_w = global_avg_pool(
                    feature_grad, name='feature_w_{}'.format(idx))
                self._feature_w_list.append(feature_w)

    def get_visualization(self, class_id=None):
        assert class_id is not None, 'class_id cannot be None!'

        with tf.name_scope('grad_cam'):
            self._comp_feature_importance_weight(class_id)
            conv_out = self._conv_out
            conv_c = tf.shape(conv_out)[-1]
            conv_h = tf.shape(conv_out)[1]
            conv_w = tf.shape(conv_out)[2]
            conv_reshape = tf.reshape(conv_out, [conv_h * conv_w, conv_c])

            o_h = tf.shape(self.input_im)[1]
            o_w = tf.shape(self.input_im)[2]

            classmap_list = []
            for idx, feature_w in enumerate(self._feature_w_list):
                feature_w = tf.reshape(feature_w, [conv_c, 1])
                classmap = tf.matmul(conv_reshape, feature_w)
                classmap = tf.reshape(classmap, [-1, conv_h, conv_w, 1])
                classmap = tf.nn.relu(
                    tf.image.resize_bilinear(classmap, [o_h, o_w]),
                    name='grad_cam_{}'.format(idx))
                classmap_list.append(tf.squeeze(classmap))

            return classmap_list, tf.convert_to_tensor(class_id)


class ClassifyGradCAM(BaseGradCAM):
    def _create_model(self, inputs):
        keep_prob = 1
        self._vis_model.create_model([inputs, keep_prob])

    def setup_graph(self):
        self.input_im = self._vis_model.layer['input']
        self._out_act = global_avg_pool(self._vis_model.layer['output'])
        self._conv_out = self._vis_model.layer['conv_out']
        self._nclass = self._out_act.shape.as_list()[-1]
        self.pre_label = tf.nn.top_k(tf.nn.softmax(self._out_act),
                                     k=5, sorted=True)
