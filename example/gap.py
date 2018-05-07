#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gap.py
# Author: Qian Ge <geqian1001@gmail.com>

import argparse
import numpy as np
import tensorflow as tf
# from tensorcv.dataflow.dataset.CIFAR import CIFAR
import sys
sys.path.append('../')
from lib.dataflow.cifar import CIFAR
from lib.models.gap import GAPNet

# data_path = '/Users/gq/workspace/Dataset/cifar-10-batches-py/'
data_path = '/home/qge2/workspace/data/dataset/cifar/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0005, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = get_args()
    dataset = CIFAR(data_dir=data_path, batch_dict_name=['im', 'label'])
    dataset.setup(epoch_val=0, batch_size=128)

    im = tf.placeholder(tf.float32, [None, 32, 32, 3], name='im')
    label = tf.placeholder(tf.int64, [None], name='label')
    input_dict = {'input': im, 'label': label}

    model = GAPNet(num_class=10, pre_train_path=pre_train_path)
    model.create_model(input_dict)

    train_op = model.get_train_op()
    loss_op = model.get_loss()
    accuracy_op = model.get_accuracy()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_sum = 0
        acc_sum = 0
        for step in range(0, 1000000):
            batch_data = dataset.next_batch_dict()
            _, loss, acc = sess.run([train_op, loss_op, accuracy_op],
                               feed_dict={model.lr: FLAGS.lr,
                                          im: batch_data['im'],
                                          label: batch_data['label']})
            loss_sum += loss
            acc_sum += acc
            if step % 20 == 0:
                print('loss: {}, acc: {}'\
                      .format(loss_sum * 1.0 / 20,
                              acc_sum * 1.0 / 20))
                loss_sum = 0
                acc_sum = 0
