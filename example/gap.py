#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gap.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import numpy as np
import tensorflow as tf
# from tensorcv.dataflow.dataset.CIFAR import CIFAR
import sys
sys.path.append('../')
from lib.dataflow.cifar import CIFAR
from lib.models.gap import GAPNet
import lib.utils.viz as viz
import lib.utils.normalize as normlize

# data_path = '/Users/gq/workspace/Dataset/cifar-10-batches-py/'
# save_path = '/Users/gq/workspace/Tmp/test/'

data_path = '/home/qge2/workspace/data/dataset/cifar/'
save_path = '/home/qge2/workspace/data/out/gap/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--epoch', default=150, type=int)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--viz', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = get_args()
    max_epoch = FLAGS.epoch
    lr = FLAGS.lr
    dropout = FLAGS.dropout

    train_data = CIFAR(data_dir=data_path,
                       batch_dict_name=['im', 'label'],
                       data_type='train',
                       substract_mean=False)
    train_data.setup(epoch_val=0, batch_size=128)
    valid_data = CIFAR(data_dir=data_path,
                       shuffle=False,
                       batch_dict_name=['im', 'label'],
                       data_type='valid',
                       # channel_mean=train_data.channel_mean,
                       substract_mean=False)
    valid_data.setup(epoch_val=0, batch_size=128)

    # print(train_data.next_batch_dict())

    im = tf.placeholder(tf.float32, [None, 32, 32, 3], name='im')
    label = tf.placeholder(tf.int64, [None], name='label')
    input_dict = {'input': im, 'label': label}

    model = GAPNet(num_class=10, wd=FLAGS.wd)
    model.create_model(input_dict)

    train_op = model.get_train_op()
    loss_op = model.get_loss()
    accuracy_op = model.get_accuracy()

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.viz:
            saver.restore(sess, '{}epoch_{}'.format(save_path, 69))
            valid_data.setup(epoch_val=0, batch_size=50)
            batch_data = valid_data.next_batch_dict()
            maps = sess.run(model.layer['feature'],
                           feed_dict={im: batch_data['im']})
            print(batch_data['label'])

            viz.viz_filters(
                batch_data['im'].transpose(1, 2, 3, 0),
                [10, 1],
                os.path.join(save_path, 'im.png'),
                gap=2,
                gap_color=10,
                shuffle=False,
                # nf=normlize.norm_range
                )

            for idx, cur_map in enumerate(maps):
                viz.viz_filters(
                    cur_map,
                    [1, 10],
                    os.path.join(save_path, 'maps_{}.png'.format(idx)),
                    gap=2,
                    gap_color=10,
                    shuffle=False,
                    # nf=normlize.norm_range
                    )

        if FLAGS.train:
            loss_sum = 0
            acc_sum = 0
            epoch_id = 0
            # for epoch_id in range(0, max_epoch):
            epoch_step = 0
            while epoch_id < max_epoch:
                epoch_step += 1
                cur_epoch = train_data.epochs_completed
                if epoch_step % int(train_data.batch_step / 10) == 0:
                    print('loss: {}, acc: {}'\
                          .format(
                                  loss_sum * 1.0 / epoch_step,
                                  acc_sum * 1.0 / epoch_step))
                if cur_epoch > epoch_id:
                    saver.save(sess, '{}epoch_{}'.format(save_path, epoch_id))
                    print('epoch: {}, lr: {}, loss: {}, acc: {}'\
                          .format(epoch_id,
                                  lr,
                                  loss_sum * 1.0 / epoch_step,
                                  acc_sum * 1.0 / epoch_step))
                    loss_sum = 0
                    acc_sum = 0
                    epoch_step = 0
                    epoch_id = cur_epoch

                    if cur_epoch >= 50:
                        lr = FLAGS.lr / 10
                    if cur_epoch >= 100:
                        lr = FLAGS.lr / 100

                    model.set_is_training(False)
                    valid_acc_sum = 0
                    valid_step = 0
                    while valid_data.epochs_completed < 1:
                        valid_step += 1
                        batch_data = valid_data.next_batch_dict()
                        acc = sess.run(accuracy_op,
                                   feed_dict={model.dropout: 1.0,
                                              im: batch_data['im'],
                                              label: batch_data['label'],})
                        valid_acc_sum += acc
                    print('valid acc: {}'.format(valid_acc_sum * 1.0 / valid_step))
                    model.set_is_training(True)
                    valid_data.setup(epoch_val=0, batch_size=128)


                batch_data = train_data.next_batch_dict()
                _, loss, acc = sess.run([train_op, loss_op, accuracy_op],
                                   feed_dict={model.lr: lr,
                                              model.dropout: dropout,
                                              im: batch_data['im'],
                                              label: batch_data['label']})
                loss_sum += loss
                acc_sum += acc
