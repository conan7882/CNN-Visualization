# File: model.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

import tensorcv
from tensorcv.models.layers import *
from tensorcv.models.base import BaseModel

from tensorcv.utils.common import deconv_size

class mnistCAM(BaseModel):
    """ training model """
    def __init__(self, num_class = 10, 
                 inspect_class = None,
                 num_channels = 1, 
                 im_height = 28, im_width = 28,
                 learning_rate = 0.0001):

        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.im_height = im_height
        self.im_width = im_width
        self.num_class = num_class
        self._inspect_class = inspect_class

        self.set_is_training(True)

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(tf.float32, name = 'image',
                            shape = [None, self.im_height, self.im_width, self.num_channels])
        self.label = tf.placeholder(tf.int64, [None], 'label')
        # self.label = tf.placeholder(tf.int64, [None, self.num_class], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob = 0.5)
        self.set_train_placeholder([self.image, self.label])
        # self.set_prediction_placeholder([self.image, self.label])
        self.set_prediction_placeholder(self.image)

    def _create_conv(self, input_im):
        conv1 = conv(input_im, 5, 32, 'conv1', nl = tf.nn.relu)
        pool1 = max_pool(conv1, 'pool1', padding = 'VALID')

        conv2 = conv(pool1, 5, 64, 'conv2', nl = tf.nn.relu)
        pool2 = max_pool(conv2, 'pool2', padding = 'VALID')

        return pool2

    def _create_model(self):

        input_im = self.model_input[0]
        keep_prob = self.model_input[1]

        conv_out = self._create_conv(input_im)
        # conv_cam = conv(conv_out, 5, 128, 'conv_cam', nl = tf.nn.relu)

        gap = global_avg_pool(conv_out)
        # dropout_gap = dropout(gap, keep_prob, self.is_training)

        with tf.variable_scope('fc1'):
            fc_w = tf.get_variable('weights', shape=[64, self.num_class], initializer=tf.random_normal_initializer(0., 0.01))
            fc1 = tf.matmul(gap, fc_w, name = 'output')

        self.output = tf.identity(fc1, 'model_output') 
        self.prediction = tf.argmax(fc1, name = 'pre_label', axis = -1)
        self.prediction_pro = tf.nn.softmax(fc1, name = 'pre_pro')

    def _get_loss(self):
        with tf.name_scope('loss'):
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                    (logits = self.output, labels = self.label), 
                    name = 'result') 

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(beta1=0.5, learning_rate = self.learning_rate)

    def _ex_setup_graph(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.prediction, self.label)
            self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, tf.float32), 
                        name = 'result')

    def _setup_summary(self):
        tf.summary.scalar("train_accuracy", self.accuracy, collections = ['train'])



class mnistCAMTest(mnistCAM):
    def _ex_setup_graph(self):
        pass

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(tf.float32, name = 'image',
                            shape = [None, None, None, self.num_channels])
        self.label = tf.placeholder(tf.int64, [None], 'label')
        # self.label = tf.placeholder(tf.int64, [None, self.num_class], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob = 0.5)
        self.set_train_placeholder([self.image, self.label])
        # self.set_prediction_placeholder([self.image, self.label])
        self.set_prediction_placeholder(self.image)

    def _create_model(self):

        input_im = self.model_input[0]
        keep_prob = self.model_input[1]

        conv_out = self._create_conv(input_im)
        conv_cam = conv(conv_out, 5, 128, 'conv3', nl = tf.nn.relu)

        gap = global_avg_pool(conv_cam)
        dropout_gap = dropout(gap, keep_prob, self.is_training)

        with tf.variable_scope('fc1'):
            fc_w = tf.get_variable('weights', shape=[128, self.num_class], initializer=tf.random_normal_initializer(0., 0.01))
        #     fc1 = tf.matmul(dropout_gap, fc_w, name = 'output')

        # self.output = tf.identity(fc1, 'model_output') 
        # self.prediction = tf.argmax(fc1, name = 'pre_label', axis = -1)
        # self.prediction_pro = tf.nn.softmax(fc1, name = 'pre_pro')

        if self._inspect_class is not None:
            self.get_classmap(self._inspect_class, conv_cam, input_im) 

    def get_classmap(self, label, conv_out, input_im):
        """
        Compute class activation map of class = label with name 'classmap'

        Args:
            label (int): a scalar int indicate the class label
            conv_out (tf.tensor): 4-D Tensor of shape 
                             [batch, height, width, channels]. 
                             Output of convolutional layers.
            input_im (tf.tensor): A 4-D Tensor image. 
                         The original model input image patch.      
        """
        # Get original image size used for interpolation
        o_height = tf.shape(input_im)[1]
        o_width = tf.shape(input_im)[2]
        # Get number of channels of output of convolution layers
        conv_out_channel = tf.shape(conv_out)[-1]

        # Interpolate to orginal size
        conv_resized = tf.image.resize_bilinear(conv_out, [o_height, o_width])

        # Get weights corresponding to class = label
        with tf.variable_scope('fc1') as scope:
            scope.reuse_variables()
            label_w = tf.gather(tf.transpose(tf.get_variable('weights')), label)
            label_w = tf.reshape(label_w, [-1, conv_out_channel, 1]) 
            label_w = tf.tile(label_w, [tf.shape(conv_out)[0], 1, 1])

        conv_resized = tf.reshape(conv_resized, [-1, o_height * o_width, conv_out_channel])

        classmap = tf.matmul(conv_resized, label_w)
        classmap = tf.reshape(classmap, [-1, o_height, o_width, 1], name = 'classmap')
        # Set negative value to be zero
        # classmap = tf.reshape(tf.nn.relu(classmap), [-1, o_height, o_width, 1], name = 'classmap')


