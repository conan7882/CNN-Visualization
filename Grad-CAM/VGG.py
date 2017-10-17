# File: VGG.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

from tensorcv.models.layers import *
from tensorcv.models.base import BaseModel


VGG_MEAN = [103.939, 116.779, 123.68]

class BaseVGG(BaseModel):
    """ base of VGG class """
    def __init__(self, num_class=1000, 
                 num_channels=3, 
                 im_height=224, im_width=224,
                 learning_rate=0.0001,
                 is_load=False,
                 pre_train_path=None):
        """ 
        Args:
            num_class (int): number of image classes
            num_channels (int): number of input channels
            im_height, im_width (int): size of input image
                               Can be unknown when testing.
            learning_rate (float): learning rate of training
        """

        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.im_height = im_height
        self.im_width = im_width
        self.num_class = num_class

        self.layer = {}

        self._is_load = is_load
        if self._is_load and pre_train_path is None:
            raise ValueError('pre_train_path can not be None!')
        self._pre_train_path = pre_train_path

        self.set_is_training(True)

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(tf.float32, name='image',
                            shape=[None, self.im_height, self.im_width, self.num_channels])
        self.label = tf.placeholder(tf.int64, [None], 'label')
        # self.label = tf.placeholder(tf.int64, [None, self.num_class], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob=0.5)
        self.set_train_placeholder([self.image, self.label])
        self.set_prediction_placeholder(self.image)

    @staticmethod
    def load_pre_trained(session, model_path, skip_layer=[]):
        weights_dict = np.load(model_path, encoding='latin1').item()
        for layer_name in weights_dict:
            print('Loading ' + layer_name)
            if layer_name not in skip_layer:
                with tf.variable_scope(layer_name, reuse=True):
                    for data in weights_dict[layer_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

class VGG19(BaseVGG):

    def _create_conv(self, input_im, data_dict):

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv], nl=tf.nn.relu, trainable=True, data_dict=data_dict):
            conv1_1 = conv(input_im, 3, 64, 'conv1_1')
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
            conv5_4 = conv(conv5_3, 3, 512, 'conv5_4')
            self.layer['conv5_4'] = conv5_4
            pool5 = max_pool(conv5_4, 'pool5', padding='SAME')

        # self.conv_out = tf.identity(conv5_4)

        return pool5

    def _create_model(self):

        input_im = self.model_input[0]
        keep_prob = self.model_input[1]

        input_im = tf.reshape(input_im, [-1, 224, 224, 3])
        # Convert RGB image to BGR image
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, 
                                    value=input_im)

        input_bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        data_dict = {}
        if self._is_load:
            data_dict = np.load(self._pre_train_path, encoding='latin1').item()

        conv_output = self._create_conv(input_bgr, data_dict)
        
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([fc], trainable=True, data_dict=data_dict):
            fc6 = fc(conv_output, 4096, 'fc6', nl=tf.nn.relu)
            dropout_fc6 = dropout(fc6, keep_prob, self.is_training)

            fc7 = fc(dropout_fc6, 4096, 'fc7', nl=tf.nn.relu)
            dropout_fc7 = dropout(fc7, keep_prob, self.is_training)

            fc8 = fc(dropout_fc7, self.num_class, 'fc8')
            self.layer['fc8'] = self.layer['output'] = fc8

        # self.output = tf.identity(fc8, 'model_output')

class VGG19_FCN(VGG19):

    def _create_model(self):

        input_im = self.model_input[0]
        keep_prob = self.model_input[1]

        # Convert rgb image to bgr image
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, 
                                    value=input_im)

        input_bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        data_dict = {}
        if self._is_load:
            data_dict = np.load(self._pre_train_path, encoding='latin1').item()

        conv_outptu = self._create_conv(input_bgr, data_dict)

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv], trainable=True, data_dict=data_dict):

            fc6 = conv(conv_outptu, 7, 4096, 'fc6', nl=tf.nn.relu, padding='VALID')
            dropout_fc6 = dropout(fc6, keep_prob, self.is_training)

            fc7 = conv(dropout_fc6, 1, 4096, 'fc7', nl=tf.nn.relu, padding='VALID')
            dropout_fc7 = dropout(fc7, keep_prob, self.is_training)

            fc8 = conv(dropout_fc7, 1, self.num_class, 'fc8', padding='VALID')
            self.layer['fc8'] = self.layer['output'] = fc8

        # self.conv_output = tf.identity(conv5_4, 'conv_output')
        self.output = tf.identity(fc8, 'model_output')
        filter_size = [tf.shape(fc8)[1], tf.shape(fc8)[2]]

        self.avg_output = global_avg_pool(fc8)

    @staticmethod
    def load_pre_trained(session, model_path, skip_layer=[]):
        fc_layers = ['fc6', 'fc7', 'fc8']
        weights_dict = np.load(model_path, encoding='latin1').item()
        for layer_name in weights_dict:
            print('Loading ' + layer_name)
            if layer_name not in skip_layer:
                with tf.variable_scope(layer_name, reuse=True):
                    for data in weights_dict[layer_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            if layer_name == 'fc6':
                                data = tf.reshape(data, [7,7,512,4096])
                            elif layer_name == 'fc7':
                                data = tf.reshape(data, [1,1,4096,4096])
                            elif layer_name == 'fc8':
                                data = tf.reshape(data, [1,1,4096,1000])
                            session.run(var.assign(data))

# if __name__ == '__main__':
#     VGG = VGG19(num_class=1000, 
#                 num_channels=3, 
#                 im_height=224, 
#                 im_width=224)
    
#     VGG.create_graph()

#     writer = tf.summary.FileWriter(config.summary_dir)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         writer.add_graph(sess.graph)

#     writer.close()



 