# File gradCAM.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from scipy import misc
import scipy.io

from tensorcv.models.layers import *
from tensorcv.dataflow.image import *
from tensorcv.utils.viz import image_overlay

import VGG

def resize_tensor_image_with_smallest_side(image, small_size):
    """
    Resize image tensor with smallest side = small_size and
    keep the original aspect ratio.

    Args:
        image (tf.tensor): 4-D Tensor of shape [batch, height, width, channels] 
            or 3-D Tensor of shape [height, width, channels].
        small_size (int): A 1-D int. The smallest side of resize image.

    Returns:
        Image tensor with the original aspect ratio and 
        smallest side = small_size .
        If images was 4-D, a 4-D float Tensor of shape 
        [batch, new_height, new_width, channels]. 
        If images was 3-D, a 3-D float Tensor of shape 
        [new_height, new_width, channels].       
    """
    im_shape = tf.shape(image)
    shape_dim = image.get_shape()
    if len(shape_dim) <= 3:
        height = tf.cast(im_shape[0], tf.float32)
        width = tf.cast(im_shape[1], tf.float32)
    else:
        height = tf.cast(im_shape[1], tf.float32)
        width = tf.cast(im_shape[2], tf.float32)

    height_smaller_than_width = tf.less_equal(height, width)

    new_shorter_edge = tf.constant(small_size, tf.float32)
    new_height, new_width = tf.cond(
    height_smaller_than_width,
    lambda: (new_shorter_edge, (width/height)*new_shorter_edge),
    lambda: ((height/width)*new_shorter_edge, new_shorter_edge))

    return tf.image.resize_images(tf.cast(image, tf.float32), 
        [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)])

class GradCAM(object):
    def __init__(self, vis_model=None, num_channel=3):
        self._vis_model = vis_model
        self._nchannel = num_channel

    def create_graph(self):
        self.image = tf.placeholder(tf.float32, name='image',
                     shape=[1, None, None, self._nchannel])
        self.in_im = resize_tensor_image_with_smallest_side(self.image, 224)
        keep_prob = 1

        self._vis_model.create_model([self.in_im, keep_prob])

        self._out_act = global_avg_pool(self._vis_model.layer['fc8'])
        self._conv_out = self._vis_model.layer['conv5_4']

        self._nclass = self._out_act.shape.as_list()[-1]
        self.pre_label = tf.nn.top_k(tf.nn.softmax(self._out_act), k=5, sorted=True)

    def comp_feature_importance_weight(self, class_id):
        if not isinstance(class_id, list):
            class_id = [class_id]

        self._feature_w_list = []
        for cid in class_id:
            one_hot = tf.sparse_to_dense([[cid, 0]], [self._nclass, 1], 1.0)
            out_act = tf.reshape(self._out_act, [1, self._nclass])
            class_act = tf.matmul(out_act, one_hot)
            feature_grad = tf.gradients(class_act, self._conv_out)
            feature_grad = tf.squeeze(tf.convert_to_tensor(feature_grad), axis = 0)
            feature_w = global_avg_pool(feature_grad)
            self._feature_w_list.append(feature_w)

    def comp_grad_cam(self, class_id=None):
        if not class_id is None:
            self.comp_feature_importance_weight(class_id)
        
        conv_out = self._conv_out
        conv_c = tf.shape(conv_out)[-1]
        conv_h = tf.shape(conv_out)[1]
        conv_w = tf.shape(conv_out)[2]
        conv_reshape = tf.reshape(conv_out, [conv_h * conv_w, conv_c])

        o_h = tf.shape(self.in_im)[1]
        o_w = tf.shape(self.in_im)[2]

        classmap_list = []
        for feature_w in self._feature_w_list:
            feature_w = tf.reshape(feature_w, [conv_c, 1])
            classmap = tf.matmul(conv_reshape, feature_w)
            classmap = tf.reshape(classmap, [-1, conv_h, conv_w, 1])
            classmap = tf.nn.relu(tf.image.resize_bilinear(classmap, [o_h, o_w]))
            classmap_list.append(tf.squeeze(classmap))
        return classmap_list


if __name__ == '__main__':
    data_dir = 'E:\\GITHUB\\workspace\\CNN\\dataset\\test\\'
    vgg_dir = 'E:\\GITHUB\\workspace\\CNN\\pretrained\\vgg19.npy'

    gcam = GradCAM(vis_model=VGG.VGG19_FCN(is_load=True, pre_train_path=vgg_dir))

    gcam.create_graph()
    # gcam.comp_feature_importance_weight([1, 2])
    map_op = gcam.comp_grad_cam(class_id=1)
    label_op = gcam.pre_label

    # im_path = 'E:\\GITHUB\\workspace\\CNN\\dataset\\ILSVRC2017_test_00000004.JPEG'
    
    # im = misc.imread(im_path, mode='RGB')
    # # im = misc.imresize(im, (224, 224, 3))
    # im = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])

    input_im = ImageFromFile('.jpg', data_dir=data_dir, 
                             num_channel=3, shuffle=False)

    writer = tf.summary.FileWriter('E:\\GITHUB\\workspace\\CNN\\test\\')
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        cnt = 0
        while input_im.epochs_completed < 1:
            im = input_im.next_batch()[0]
            map_t, label, o_im = sess.run([map_op, label_op, gcam.in_im], feed_dict={gcam.image: im})
            print(label)
            save_path = 'E:\\GITHUB\\workspace\\CNN\\test\\test_{}.png'.format(cnt)
            overlay_im = image_overlay(map_t[0], o_im)
            scipy.misc.imsave(save_path, overlay_im)
            # scipy.io.savemat(save_path, {'classmap': map_t})
            cnt += 1
    writer.close()