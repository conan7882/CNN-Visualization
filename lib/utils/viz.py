#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np


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
