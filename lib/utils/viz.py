#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import scipy.misc

import lib.utils.normalize as normlize


def image_weight_mask(image, mask):
    """
    Args:
        image: image with size [HEIGHT, WIDTH, CHANNEL]
        mask: image with size [HEIGHT, WIDTH, 1] or [HEIGHT, WIDTH]
    """
    image = np.array(np.squeeze(image))
    mask = np.array(np.squeeze(mask))
    assert len(mask.shape) == 2
    assert len(image.shape) < 4
    mask.astype('float32')
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
    mask = mask / np.amax(mask)

    if len(image.shape) == 2:
        return np.multiply(image, mask)
    else:
        for c in range(0, image.shape[2]):
            image[:, :, c] = np.multiply(image[:, :, c], mask)
        return image

# def save_merge_images(images, merge_grid, save_path, color=False, tanh=False):
#     """Save multiple images with same size into one larger image.
#     The best size number is
#     int(max(sqrt(image.shape[0]),sqrt(image.shape[1]))) + 1
#     Args:
#         images (np.ndarray): A batch of image array to be merged with size
#             [BATCH_SIZE, HEIGHT, WIDTH, CHANNEL].
#         merge_grid (list): List of length 2. The grid size for merge images.
#         save_path (str): Path for saving the merged image.
#         color (bool): Whether convert intensity image to color image.
#         tanh (bool): If True, will normalize the image in range [-1, 1]
#             to [0, 1] (for GAN models).
#     Example:
#         The batch_size is 64, then the size is recommended [8, 8].
#         The batch_size is 32, then the size is recommended [6, 6].
#     """

#     # normalization of tanh output
#     img = images

#     if tanh:
#         img = (img + 1.0) / 2.0

#     if color:
#         # TODO
#         img_list = []
#         for im in np.squeeze(img):
#             im = intensity_to_rgb(np.squeeze(im), normalize=True)
#             img_list.append(im)
#         img = np.array(img_list)
#         # img = np.expand_dims(img, 0)

#     if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] <= 4):
#         img = np.expand_dims(img, 0)
#     # img = images
#     h, w = img.shape[1], img.shape[2]
#     merge_img = np.zeros((h * merge_grid[0], w * merge_grid[1], 3))
#     if len(img.shape) < 4:
#         img = np.expand_dims(img, -1)

#     for idx, image in enumerate(img):
#         i = idx % merge_grid[1]
#         j = idx // merge_grid[1]
#         merge_img[j*h:j*h+h, i*w:i*w+w, :] = image

#     scipy.misc.imsave(save_path, merge_img)

def viz_filters(filters,
                grid_size,
                save_path,
                gap=0,
                gap_color=0,
                nf=normlize.indentity,
                shuffle=True):
    """ Visualization conv2d filters

    Args:
        filters: [size_x, size_y, n_channel, n_features]
                or [size_x, size_y, n_features]

    """
    filters = np.array(filters)
    if len(filters.shape) == 4:
        n_channel = filters.shape[2]
    elif len(filters.shape) == 3:
        n_channel = 1
        filters = np.expand_dims(filters, axis=2)
    # assert len(filters.shape) == 4
    assert len(grid_size) == 2

    h = filters.shape[0]
    w = filters.shape[1]

    merge_im = np.zeros((h * grid_size[0] + (grid_size[0] + 1) * gap,
                         w * grid_size[1] + (grid_size[1] + 1) * gap,
                         n_channel)) + gap_color

    n_viz_filter = min(filters.shape[-1], grid_size[0] * grid_size[1])
    if shuffle == True:
        pick_id = np.random.permutation(filters.shape[-1])
    else:
        pick_id = range(0, filters.shape[-1])
    for idx in range(0, n_viz_filter):
        i = idx % grid_size[1]
        j = idx // grid_size[1]
        cur_filter = filters[:, :, :, pick_id[idx]]
        merge_im[j * (h + gap) + gap: j * (h + gap) + h + gap,
                 i * (w + gap) + gap: i * (w + gap) + w + gap, :]\
            = nf(cur_filter)
    scipy.misc.imsave(save_path, np.squeeze(merge_im))











