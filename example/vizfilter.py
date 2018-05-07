#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vizfilter.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
from lib.nets.googlenet import BaseGoogLeNet
import lib.utils.viz as viz

pre_train_path = '/Users/gq/workspace/Dataset/pretrained/googlenet.npy'
save_path = '/Users/gq/tmp/viz/'

if __name__ == '__main__':

    model = BaseGoogLeNet(pre_train_path)
    filters = tf.get_default_graph().get_tensor_by_name('conv1_7x7_s2/weights:0')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        learned_filter = sess.run(filters)
        viz.viz_filters(learned_filter,
                        [8, 8],
                        os.path.join(save_path, 'test.png'),
                        gap=2)
