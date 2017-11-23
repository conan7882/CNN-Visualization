#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test_guided_backprop.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.dataflow.image import ImageFromFile

from setup_test_env import *
from nets.vgg import VGG19_FCN
from models.guided_backpro import GuideBackPro


def test_guided_backprop():
    # placeholder for input image
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    # initialize input dataflow
    # change '.png' to other image types if other types of images are used
    input_im = ImageFromFile('.png', data_dir=IMPATH,
                             num_channel=3, shuffle=False)
    # batch size has to be one
    input_im.set_batch_size(1)

    # initialize guided back propagation class
    # use VGG19 as an example
    # images will be rescaled to smallest side = 224 is is_rescale=True
    model = GuideBackPro(vis_model=VGG19_FCN(is_load=False,
                                             is_rescale=True))

    # get op to compute guided back propagation map
    # final output respect to input image
    back_pro_op = model.get_visualization(image)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        im = input_im.next_batch()[0]
        guided_backpro, label, o_im =\
            sess.run([back_pro_op, model.pre_label,
                     model.input_im],
                     feed_dict={image: im})
        print(label)
    tf.reset_default_graph()

