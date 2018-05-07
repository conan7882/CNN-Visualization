#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vizfilter.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import tensorflow as tf

import config_path as config

import sys
sys.path.append('../')
from lib.nets.googlenet import BaseGoogLeNet
import lib.utils.viz as viz


if __name__ == '__main__':

    model = BaseGoogLeNet(config.googlenet_path)
    filters = tf.get_default_graph().get_tensor_by_name(
        'conv1_7x7_s2/weights:0')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        learned_filter = sess.run(filters)
        viz.viz_filters(learned_filter,
                        [8, 8],
                        os.path.join(config.save_path, 'GoogLeNet.png'),
                        gap=2)
