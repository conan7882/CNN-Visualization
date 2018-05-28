#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: deconv.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import scipy
import argparse
import tensorflow as tf

import config_path as config

import sys
sys.path.append('../')
from lib.nets.vgg import DeconvBaseVGG19, BaseVGG19
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
    
    # map_list = ['inception4a', 'inception4b', 'inception4c',
    #             'inception4d', 'inception4e', 'inception3a',
    #             'inception3b', 'inception5a', 'inception5b']

    model = BaseVGG19(config.vgg_path)
    vizmodel = DeconvBaseVGG19(config.vgg_path)

    feats = model.conv_layer['conv2_2']

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
        
