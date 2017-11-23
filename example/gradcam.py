#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gradcam.py
# Author: Qian Ge <geqian1001@gmail.com>

from itertools import count

import tensorflow as tf
import numpy as np
from tensorcv.dataflow.image import ImageFromFile
from tensorcv.utils.viz import image_overlay, save_merge_images

import setup_env
from nets.vgg import VGG19_FCN
from models.guided_backpro import GuideBackPro
from models.grad_cam import ClassifyGradCAM
from utils.viz import image_weight_mask

IM_PATH = '../data/'
SAVE_DIR = '../../data/tmp/'
VGG_PATH = '../../data/pretrain/vgg/vgg19.npy'


# def image_weight_mask(image, mask):
#     """
#     Args:
#         image: image with size [HEIGHT, WIDTH, CHANNEL]
#         mask: image with size [HEIGHT, WIDTH, 1] or [HEIGHT, WIDTH]
#     """
#     image = np.array(np.squeeze(image))
#     mask = np.array(np.squeeze(mask))
#     assert len(mask.shape) == 2
#     assert len(image.shape) < 4
#     mask.astype('float32')
#     mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
#     mask = mask / np.amax(mask)

#     if len(image.shape) == 2:
#         return np.multiply(image, mask)
#     else:
#         for c in range(0, image.shape[2]):
#             image[:, :, c] = np.multiply(image[:, :, c], mask)
#         return image


if __name__ == '__main__':

    # merge several output images in one large image
    merge_im = 1
    grid_size = np.ceil(merge_im**0.5).astype(int)

    # class label for Grad-CAM generation
    # 355 llama 543 dumbbell 605 iPod 515 hat 99 groose 283 tiger cat
    # 282 tabby cat 233 border collie 242 boxer
    # class_id = [355, 543, 605, 515]
    class_id = [283, 242]

    # initialize Grad-CAM
    # using VGG19
    gcam = ClassifyGradCAM(
        vis_model=VGG19_FCN(is_load=True,
                            pre_train_path=VGG_PATH,
                            is_rescale=True))
    gbackprob = GuideBackPro(
        vis_model=VGG19_FCN(is_load=True,
                            pre_train_path=VGG_PATH,
                            is_rescale=True))

    # placeholder for input image
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3])

    # create VGG19 model
    gcam.create_model(image)
    gcam.setup_graph()

    # generate class map and prediction label ops
    map_op = gcam.get_visualization(class_id=class_id)
    label_op = gcam.pre_label

    back_pro_op = gbackprob.get_visualization(image)

    # initialize input dataflow
    # change '.png' to other image types if other types of images are used
    input_im = ImageFromFile('.png', data_dir=IM_PATH,
                             num_channel=3, shuffle=False)
    input_im.set_batch_size(1)

    writer = tf.summary.FileWriter(SAVE_DIR)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        cnt = 0
        merge_cnt = 0
        # weight_im_list = [[] for i in range(len(class_id))]
        o_im_list = []
        while input_im.epochs_completed < 1:
            im = input_im.next_batch()[0]
            gcam_map, b_map, label, o_im =\
                sess.run([map_op, back_pro_op, label_op, gcam.input_im],
                         feed_dict={image: im})
            print(label)
            o_im_list.extend(o_im)
            for idx, cid, cmap in zip(count(), gcam_map[1], gcam_map[0]):
                overlay_im = image_overlay(cmap, o_im)
                weight_im = image_weight_mask(b_map[0], cmap)
                try:
                    weight_im_list[idx].append(weight_im)
                    overlay_im_list[idx].append(overlay_im)
                except NameError:
                    gcam_class_id = gcam_map[1]
                    weight_im_list = [[] for i in range(len(gcam_class_id))]
                    overlay_im_list = [[] for i in range(len(gcam_class_id))]
                    weight_im_list[idx].append(weight_im)
                    overlay_im_list[idx].append(overlay_im)
            merge_cnt += 1

            # Merging results
            if merge_cnt == merge_im:
                save_path = '{}oim_{}.png'.format(SAVE_DIR, cnt, cid)
                save_merge_images(np.array(o_im_list),
                                  [grid_size, grid_size],
                                  save_path)
                for w_im, over_im, cid in zip(weight_im_list,
                                              overlay_im_list,
                                              gcam_class_id):
                    # save grad-cam results
                    save_path = '{}gradcam_{}_class_{}.png'.\
                        format(SAVE_DIR, cnt, cid)
                    save_merge_images(
                        np.array(over_im), [grid_size, grid_size], save_path)
                    # save guided grad-cam results
                    save_path = '{}guided_gradcam_{}_class_{}.png'.\
                        format(SAVE_DIR, cnt, cid)
                    save_merge_images(
                        np.array(w_im), [grid_size, grid_size], save_path)
                weight_im_list = [[] for i in range(len(gcam_class_id))]
                overlay_im_list = [[] for i in range(len(gcam_class_id))]
                o_im_list = []
                merge_cnt = 0
                cnt += 1

        # Saving results
        if merge_cnt > 0:
            save_path = '{}oim_{}.png'.format(SAVE_DIR, cnt, cid)
            save_merge_images(np.array(o_im_list),
                              [grid_size, grid_size],
                              save_path)
            for w_im, over_im, cid in zip(weight_im_list,
                                          overlay_im_list,
                                          gcam_class_id):
                # save grad-cam results
                save_path = '{}gradcam_{}_class_{}.png'.\
                    format(SAVE_DIR, cnt, cid)
                save_merge_images(
                    np.array(over_im), [grid_size, grid_size], save_path)
                # save guided grad-cam results
                save_path = '{}guided_gradcam_{}_class_{}.png'.\
                    format(SAVE_DIR, cnt, cid)
                save_merge_images(
                    np.array(w_im), [grid_size, grid_size], save_path)
    writer.close()
