#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tensorcv.models.layers import *


@add_arg_scope
def transpose_conv(x,
                   filter_size,
                   out_dim,
                   data_dict,
                   reuse=False,
                   stride=2,
                   padding='SAME',
                   # init_w=None,
                   # init_b=None,
                   # wd=None,
                   trainable=False,
                   nl=tf.identity,
                   name='dconv'):

    stride = get_shape4D(stride)

    in_dim = x.get_shape().as_list()[-1]

    # TODO other ways to determine the output shape 
    x_shape = tf.shape(x)
    # assume output shape is input_shape*stride
    out_shape = tf.stack([x_shape[0],
                          tf.multiply(x_shape[1], stride[1]), 
                          tf.multiply(x_shape[2], stride[2]),
                          out_dim])

    filter_shape = get_shape2D(filter_size) + [out_dim, in_dim]

    with tf.variable_scope(name) as scope:
        if reuse == True:
            scope.reuse_variables()
            init_w = None
            init_b = None
        else:
            try:
                load_data = data_dict[name][0]
            except KeyError:
                load_data = data_dict[name]['weights']
            print('Load {} weights!'.format(name))
            # load_data = np.reshape(load_data, shape)
            init_w = tf.constant_initializer(load_data)

            try:
                load_data = data_dict[name][1]
            except KeyError:
                load_data = data_dict[name]['biases']
            print('Load {} biases!'.format(name))
            init_b = tf.constant_initializer(load_data)

        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable)

        biases = tf.get_variable('biases',
                                 [out_dim],
                                 initializer=init_b,
                                 trainable=trainable)

        dconv = tf.nn.conv2d_transpose(x,
                                       weights, 
                                       output_shape=out_shape, 
                                       strides=stride, 
                                       padding=padding, 
                                       name=scope.name)
        bias = tf.nn.bias_add(dconv, -biases)
        # TODO need test
        bias.set_shape([None, None, None, out_dim])

        output = nl(bias, name='output')
        return output