#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: deconv.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import scipy.misc
import argparse
import numpy as np
import tensorflow as tf
from tensorcv.dataflow.image import ImageFromFile

import config_path as config

import sys
sys.path.append('../')
from lib.nets.vgg import DeconvBaseVGG19, BaseVGG19
import lib.utils.viz as viz
import lib.utils.normalize as normlize
import lib.utils.image as uim


IM_SIZE = 224

def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imtype', type=str, default='.jpg',
                        help='Image type')
    parser.add_argument('--feat', type=str, required=True,
                        help='Choose of feature map layer')
    parser.add_argument('--id', type=int, default=None,
                        help='feature map id')

    return parser.parse_args()

def im_scale(im):
    return uim.im_rescale(im, [IM_SIZE, IM_SIZE])

if __name__ == '__main__':
    FLAGS = get_parse()
    
    input_im = ImageFromFile(FLAGS.imtype,
                             data_dir=config.im_path,
                             num_channel=3,
                             shuffle=False,
                             pf=im_scale,
                             )
    input_im.set_batch_size(1)

    vizmodel = DeconvBaseVGG19(config.vgg_path,
                               feat_key=FLAGS.feat,
                               pick_feat=FLAGS.id)

    vizmap = vizmodel.deconv_layer['deconvim']
    feat_op = vizmodel.feats
    max_act_op = vizmodel.max_act

    act_size = vizmodel.receptive_size[FLAGS.feat]
    act_scale = vizmodel.stride[FLAGS.feat]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        max_act_list = []
        while input_im.epochs_completed < 1:
            im = input_im.next_batch()[0]
            max_act = sess.run(max_act_op, feed_dict={vizmodel.im: im})
            max_act_list.append(max_act)

        max_list = np.argsort(max_act_list)[::-1]
        im_file_list = input_im.get_data_list()[0]

        feat_list = []
        im_list = []
        for i in range(0, 10):
            im = input_im.next_batch()[0]
            file_path = os.path.join(config.im_path, im_file_list[max_list[i]])
            im = np.array([im_scale(scipy.misc.imread(file_path, mode='RGB'))])

            cur_vizmap, feat_map, max_act = sess.run(
                [vizmap, feat_op, max_act_op], feed_dict={vizmodel.im: im})

            act_ind = np.nonzero((feat_map))
            print('Location of max activation {}'.format(act_ind))
            # get only the first nonzero element
            act_c = (act_ind[1][0], act_ind[2][0])
            min_x = max(0, int(act_c[0] * act_scale - act_size / 2))
            max_x = min(IM_SIZE, int(act_c[0] * act_scale + act_size / 2))
            min_y = max(0, int(act_c[1] * act_scale - act_size / 2))
            max_y = min(IM_SIZE, int(act_c[1] * act_scale + act_size / 2))

            im_crop = im[0, min_x:max_x, min_y:max_y, :]
            act_crop = cur_vizmap[0, min_x:max_x, min_y:max_y, :]

            pad_size = (act_size - im_crop.shape[0], act_size - im_crop.shape[1])
            im_crop = np.pad(im_crop,
                             ((0, pad_size[0]), (0, pad_size[1]), (0, 0)),
                             'constant',
                             constant_values=0)
            act_crop = np.pad(act_crop,
                              ((0, pad_size[0]),(0, pad_size[1]), (0, 0)),
                              'constant',
                              constant_values=0)

            feat_list.append(act_crop)
            im_list.append(im_crop)

        viz.viz_filters(np.transpose(feat_list, (1, 2, 3, 0)),
                        [3, 3],
                        os.path.join(config.save_path, '{}_feat.png'.format(FLAGS.feat)),
                        gap=2,
                        gap_color=0,
                        nf=normlize.indentity,
                        shuffle=False)
        viz.viz_filters(np.transpose(im_list, (1, 2, 3, 0)),
                        [3, 3],
                        os.path.join(config.save_path, '{}_im.png'.format(FLAGS.feat)),
                        gap=2,
                        gap_color=0,
                        nf=normlize.indentity,
                        shuffle=False)
        
