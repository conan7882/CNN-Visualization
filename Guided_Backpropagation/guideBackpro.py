# File guideBackpro.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from scipy import misc
import scipy.io

from tensorcv.models.layers import *
from tensorcv.dataflow.image import *
from tensorcv.utils.viz import image_overlay, save_merge_images

from ..Grad_CAM.VGG import VGG19_FCN

# def resize_tensor_image_with_smallest_side(image, small_size):
#     """
#     Resize image tensor with smallest side = small_size and
#     keep the original aspect ratio.

#     Args:
#         image (tf.tensor): 4-D Tensor of shape [batch, height, width, channels] 
#             or 3-D Tensor of shape [height, width, channels].
#         small_size (int): A 1-D int. The smallest side of resize image.

#     Returns:
#         Image ten sor with the original aspect ratio and 
#         smallest side = small_size .
#         If images was 4-D, a 4-D float Tensor of shape 
#         [batch, new_height, new_width, channels]. 
#         If images was 3-D, a 3-D float Tensor of shape 
#         [new_height, new_width, channels].       
#     """
#     im_shape = tf.shape(image)
#     shape_dim = image.get_shape()
#     if len(shape_dim) <= 3:
#         height = tf.cast(im_shape[0], tf.float32)
#         width = tf.cast(im_shape[1], tf.float32)
#     else:
#         height = tf.cast(im_shape[1], tf.float32)
#         width = tf.cast(im_shape[2], tf.float32)

#     height_smaller_than_width = tf.less_equal(height, width)

#     new_shorter_edge = tf.constant(small_size, tf.float32)
#     new_height, new_width = tf.cond(
#     height_smaller_than_width,
#     lambda: (new_shorter_edge, (width/height)*new_shorter_edge),
#     lambda: ((height/width)*new_shorter_edge, new_shorter_edge))

#     return tf.image.resize_images(tf.cast(image, tf.float32), 
#         [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)])

# class BaseGradCAM(object):
#     def __init__(self, vis_model=None, num_channel=3):
#         self._vis_model = vis_model
#         self._nchannel = num_channel

#     def create_graph(self):
#         raise NotImplementedError()

#     def _comp_feature_importance_weight(self, class_id):
#         if not isinstance(class_id, list):
#             class_id = [class_id]

#         with tf.name_scope('feature_weight'): 
#             self._feature_w_list = []
#             for idx, cid in enumerate(class_id):
#                 one_hot = tf.sparse_to_dense([[cid, 0]], [self._nclass, 1], 1.0)
#                 out_act = tf.reshape(self._out_act, [1, self._nclass])
#                 class_act = tf.matmul(out_act, one_hot, name='class_act_{}'.format(idx))
#                 feature_grad = tf.gradients(class_act, self._conv_out, name='grad_{}'.format(idx))
#                 feature_grad = tf.squeeze(tf.convert_to_tensor(feature_grad), axis = 0)
#                 feature_w = global_avg_pool(feature_grad, name='feature_w_{}'.format(idx))
#                 self._feature_w_list.append(feature_w)

#     def comp_grad_cam(self, class_id=None):
#         assert not class_id is None, 'class_id cannot be None!'
#         self._comp_feature_importance_weight(class_id)

#         with tf.name_scope('grad_cam'):
#             conv_out = self._conv_out
#             conv_c = tf.shape(conv_out)[-1]
#             conv_h = tf.shape(conv_out)[1]
#             conv_w = tf.shape(conv_out)[2]
#             conv_reshape = tf.reshape(conv_out, [conv_h * conv_w, conv_c])

#             o_h = tf.shape(self.in_im)[1]
#             o_w = tf.shape(self.in_im)[2]

#             classmap_list = []
#             for idx, feature_w in enumerate(self._feature_w_list):
#                 feature_w = tf.reshape(feature_w, [conv_c, 1])
#                 classmap = tf.matmul(conv_reshape, feature_w)
#                 classmap = tf.reshape(classmap, [-1, conv_h, conv_w, 1])
#                 classmap = tf.nn.relu(tf.image.resize_bilinear(classmap, [o_h, o_w]), name = 'grad_cam_{}'.format(idx))
#                 classmap_list.append(tf.squeeze(classmap))
#             return classmap_list

# class ClassifyGradCAM(BaseGradCAM):
#     def __init__(self, vis_model=None, num_channel=3, is_rescale = False):
#         self._is_rescale = is_rescale
#         super(ClassifyGradCAM, self).__init__(vis_model=vis_model, num_channel=num_channel)

#     def create_graph(self, image):

#         self.in_im = image
#         if self._is_rescale:
#             self.in_im = resize_tensor_image_with_smallest_side(image, 224)
#         keep_prob = 1

#         self._vis_model.create_model([self.in_im, keep_prob])

#         self._out_act = global_avg_pool(self._vis_model.layer['output'])
#         self._conv_out = self._vis_model.layer['conv_out']

#         self._nclass = self._out_act.shape.as_list()[-1]
#         self.pre_label = tf.nn.top_k(tf.nn.softmax(self._out_act), k=5, sorted=True)


if __name__ == '__main__':
    VGG19_FCN()
    # # setup dirs
    # data_dir = 'E:\\GITHUB\\workspace\\CNN\\dataset\\test\\'
    # vgg_dir = 'E:\\GITHUB\\workspace\\CNN\\pretrained\\vgg19.npy'
    # save_dir = 'E:\\GITHUB\\workspace\\CNN\\test\\'

    # # merge several output images in one large image
    # merge_im = 1
    # grid_size = np.ceil(merge_im**0.5).astype(int)

    # # class label for Grad-CAM generation
    # # 355 llama 543 dumbbell 605 iPod 515 hat 99 groose 283 tiger cat
    # # 282 tabby cat 233 border collie
    # class_id = [282, 233]

    # # initialize Grad-CAM 
    # # using VGG19
    # gcam = ClassifyGradCAM(vis_model=VGG.VGG19_FCN(is_load=True, pre_train_path=vgg_dir),
    #                        is_rescale = False)

    # # placeholder for input image
    # image = tf.placeholder(tf.float32, shape=[None, None, None, 3])

    # # create VGG19 model
    # gcam.create_graph(image)

    # # generate class map and prediction label ops
    # map_op = gcam.comp_grad_cam(class_id=class_id)
    # label_op = gcam.pre_label

    # # initialize input dataflow
    # input_im = ImageFromFile('.png', data_dir=data_dir, 
    #                          num_channel=3, shuffle=False, resize=224)
    # input_im.set_batch_size(1)

    # writer = tf.summary.FileWriter('E:\\GITHUB\\workspace\\CNN\\test\\')
    # with tf.Session() as sess:

    #     sess.run(tf.global_variables_initializer())
    #     writer.add_graph(sess.graph)

    #     cnt = 0
    #     merge_cnt = 0
    #     im_list = [[] for i in range(len(class_id))]
    #     o_im_list = []
    #     while input_im.epochs_completed < 1:
    #         im = input_im.next_batch()[0]
    #         map_t, label, o_im = sess.run([map_op, label_op, gcam.in_im], feed_dict={image: im})
    #         print(label)
    #         o_im_list.extend(o_im)
    #         for cid in range(len(map_t)):
    #             overlay_im = image_overlay(map_t[cid], o_im)
    #             im_list[cid].append(overlay_im)
    #         merge_cnt += 1
    #         if merge_cnt == merge_im:
    #             save_path = '{}test_oim_{}.png'.format(save_dir, cnt, cid)
    #             save_merge_images(np.array(o_im_list), [grid_size, grid_size], save_path)
    #             for im, cid in zip(im_list, class_id):
    #                 save_path = '{}test_{}_c_{}.png'.format(save_dir, cnt, cid)
    #                 save_merge_images(np.array(im), [grid_size, grid_size], save_path)
    #             im_list = [[] for i in range(len(class_id))]
    #             o_im_list = []
    #             merge_cnt = 0
    #             cnt += 1

    #     if merge_cnt > 0:
    #         save_path = '{}test_oim_{}.png'.format(save_dir, cnt, cid)
    #         save_merge_images(np.array(o_im_list), [grid_size, grid_size], save_path)
    #         for im, cid in zip(im_list, class_id):
    #             save_path = '{}test_{}_c_{}.png'.format(save_dir, cnt, cid)
    #             save_merge_images(np.array(im), [grid_size, grid_size], save_path)
    # writer.close()
