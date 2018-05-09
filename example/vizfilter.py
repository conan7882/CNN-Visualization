#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vizfilter.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import scipy
import argparse
import tensorflow as tf

import config_path as config

import sys
sys.path.append('../')
from lib.nets.googlenet import BaseGoogLeNet
import lib.utils.viz as viz
import lib.utils.normalize as normlize



def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', action='store_true',
                        help='Visualize filters')
    parser.add_argument('--feature', action='store_true',
                        help='Visualize feature maps')

    parser.add_argument('--im', type=str,
                        help='Image file name')

    return parser.parse_args()


if __name__ == '__main__':
    FLAGES = get_parse()
    
    map_list = ['inception4a', 'inception4b', 'inception4c',
                'inception4d', 'inception4e', 'inception3a',
                'inception3b', 'inception5a', 'inception5b']

    model = BaseGoogLeNet(config.googlenet_path)
    filters = tf.get_default_graph().get_tensor_by_name(
        'conv1_7x7_s2/weights:0')

    if FLAGES.feature:
        feature_map = []
        for c_map in map_list:
            feature_map.append(model.conv_layer[c_map])
        assert FLAGES.im is not None, 'File name cannot be None!'
        file_path = os.path.join(config.im_path, FLAGES.im)
        assert os.path.isfile(file_path),\
            'File does not exist! {}'.format(file_path)
        im = scipy.misc.imread(file_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGES.filter:
            learned_filter = sess.run(filters)
            viz.viz_filters(
                learned_filter,
                [8, 8],
                os.path.join(config.save_path, 'GoogLeNet_filter.png'),
                gap=2,
                nf=normlize.norm_std)

        if FLAGES.feature:
            maps = sess.run(feature_map, feed_dict={model.inputs: [im]})

            for key, c_map in zip(map_list, maps):
                viz.viz_filters(
                    c_map[0],
                    [10, 10],
                    os.path.join(config.save_path, 'GoogLeNet_{}.png'.format(key)),
                    gap=2,
                    gap_color=10,
                    # nf=normlize.norm_range
                    )
