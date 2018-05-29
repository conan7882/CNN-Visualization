#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: invert.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf 
import numpy as np

from tensorcv.models.base import BaseModel


class InvertCNN(BaseModel):
    def __init__(self, im_h, im_w, im_c, input_mean=0, input_std=1.0, mean_list=None):
        init = tf.random_normal([1, im_h, im_w, im_c])
        self.invert_im = tf.get_variable('invert_im',
                                          initializer=init,
                                          # shape=[1, im_h, im_w, im_c],
                                          trainable=True)


        self._mean = mean_list
        self._input_std = input_std

    def _total_variation(self, image):
        var_x = tf.pow(image[:, 1:, :-1, :] - image[:, :-1, :-1, :], 2)
        var_y = tf.pow(image[:, :-1, 1:, :] - image[:, :-1, :-1, :], 2)
        return tf.reduce_sum(var_x + var_y)

    def get_loss(self, feat_invert, feat_im):
        self.mse_loss = 5e-4 * tf.losses.mean_squared_error(feat_invert, feat_im)
        self.vt_loss = 0.0000005 * self._total_variation(self.invert_im)
        self.loss = 1000 * self.mse_loss + 0*self.vt_loss
        return self.loss

    def optimize_image(self, feat_invert, feat_im):
        loss = self.get_loss(feat_invert, feat_im)
        # opt = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        opt = tf.train.AdamOptimizer(learning_rate=0.1)
        return opt.minimize(loss)

    def get_opt_im(self):
        im = self.invert_im
        # if self._mean is not None:
        #     im = self._add_mean(im)
        return im