# File gradCAM.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from scipy import misc
import scipy.io

from tensorcv.models.layers import *

import VGG


class GradCAM(object):
    def __init__(self, vis_model=None, num_channel=3):
        self._vis_model = vis_model
        self._nchannel = num_channel

    def create_graph(self):
        self.image = tf.placeholder(tf.float32, name='image',
                     shape=[1, 224, 224, self._nchannel])
        keep_prob = 1

        data_dict = {}
        # pre_train_path = 'D:\\Qian\\GitHub\\workspace\\VGG\\vgg19.npy'
        pre_train_path = 'E:\\GITHUB\\workspace\\CNN\\pretrained\\vgg19.npy'
        data_dict = np.load(pre_train_path, encoding='latin1').item()
        # print(data_dict)

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv], nl=tf.nn.relu, trainable=True, data_dict=data_dict):
            conv1_1 = conv(self.image , 3, 64, 'conv1_1')
            conv1_2 = conv(conv1_1, 3, 64, 'conv1_2')
            pool1 = max_pool(conv1_2, 'pool1', padding='SAME')

            conv2_1 = conv(pool1, 3, 128, 'conv2_1')
            conv2_2 = conv(conv2_1, 3, 128, 'conv2_2')
            pool2 = max_pool(conv2_2, 'pool2', padding='SAME')

            conv3_1 = conv(pool2, 3, 256, 'conv3_1')
            conv3_2 = conv(conv3_1, 3, 256, 'conv3_2')
            conv3_3 = conv(conv3_2, 3, 256, 'conv3_3')
            conv3_4 = conv(conv3_3, 3, 256, 'conv3_4')
            pool3 = max_pool(conv3_4, 'pool3', padding='SAME')

            conv4_1 = conv(pool3, 3, 512, 'conv4_1')
            conv4_2 = conv(conv4_1, 3, 512, 'conv4_2')
            conv4_3 = conv(conv4_2, 3, 512, 'conv4_3')
            conv4_4 = conv(conv4_3, 3, 512, 'conv4_4')
            pool4 = max_pool(conv4_4, 'pool4', padding='SAME')

            conv5_1 = conv(pool4, 3, 512, 'conv5_1')
            conv5_2 = conv(conv5_1, 3, 512, 'conv5_2')
            conv5_3 = conv(conv5_2, 3, 512, 'conv5_3')
            self.conv5_4 = conv(conv5_3, 3, 512, 'conv5_4')
            pool5 = max_pool(self.conv5_4, 'pool5', padding='SAME')

        with arg_scope([conv], trainable=True, data_dict=data_dict):

            fc6 = conv(pool5, 7, 4096, 'fc6', nl=tf.nn.relu, padding='VALID')
            # dropout_fc6 = dropout(fc6, keep_prob, self.is_training)

            fc7 = conv(fc6, 1, 4096, 'fc7', nl=tf.nn.relu, padding='VALID')
            # dropout_fc7 = dropout(fc7, keep_prob, self.is_training)

            self.fc8 = conv(fc7, 1, 1000, 'fc8', padding='VALID')
            self.pre_label = tf.nn.top_k(tf.nn.softmax(self.fc8), 
                            k=5, sorted=True)



        # self.image = tf.placeholder(tf.float32, name='image',
  #                    shape=[None, None, None, self._nchannel])
        # self._vis_model.create_model([self.image, 1])

    def comp_grad_cam(self, class_id):
        one_hot = tf.sparse_to_dense([[0,class_id]], [1, 1000], 1.0)
        print(one_hot)
        # one_hot = tf.reshape(one_hot, [-1, 1000, 1])
        out = tf.reshape(self.fc8, [1000, 1])
        class_act = tf.matmul(one_hot, out)
        feature_grad = tf.gradients(class_act, self.conv5_4)
        # print(class_act)
        # print(self.conv5_4)
        # print(feature_grad)
        feature_grad = tf.squeeze(tf.convert_to_tensor(feature_grad), axis = 0)
        # print(feature_grad)
        feature_w = global_avg_pool(feature_grad)
        feature_w = tf.reshape(feature_w, [512, 1])
        # print(feature_w)

        conv_out = self.conv5_4
        conv_reshape = tf.reshape(conv_out, 
                           [14 * 14, 512])
        classmap = tf.matmul(conv_reshape, feature_w)
        classmap = tf.reshape(classmap, [1, 14, 14, 1])

        classmap = tf.nn.relu(tf.image.resize_bilinear(classmap, [224, 224], name='result'))
        return tf.squeeze(classmap, axis = 0)


if __name__ == '__main__':

    gcam = GradCAM(vis_model=VGG.VGG19(is_load=False, 
                         # pre_train_path='E:\\GITHUB\\workspace\\CNN\\pretrained\\vgg19.npy')

                        pre_train_path = 'D:\\Qian\\GitHub\\workspace\\VGG\\vgg19.npy')

    )

    gcam.create_graph()
    map_op = gcam.comp_grad_cam(1)
    label_op = gcam.pre_label

    im_path = 'E:\\GITHUB\\workspace\\CNN\\dataset\\ILSVRC2017_test_00000004.JPEG'
    save_path = 'E:\\GITHUB\\workspace\\CNN\\test\\test.mat'
    im = misc.imread(im_path, mode='RGB')
    im = misc.imresize(im, (224, 224, 3))
    im = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        map_t, label = sess.run([map_op, label_op], feed_dict = {gcam.image: im})

        print(label)
        

        scipy.io.savemat(save_path, {'classmap': map_t})