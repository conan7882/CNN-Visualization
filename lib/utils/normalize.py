#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: normalize.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np

def indentity(filter_in):
	return filter_in

def norm_std(filter_in):
    """ Normalization of conv2d filters for visualization
    https://github.com/jacobgil/keras-filter-visualization/blob/master/utils.py

    Args:
        filter_in: [size_x, size_y, n_channel]

    """
    x = filter_in
    x -= x.mean()
    x /= (x.std() + 1e-5)
    # make most of the value between [-0.5, 0.5]
    x *= 0.1
    # move to [0, 1]
    x += 0.5
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def norm_range(filter_in):
    f_min = np.amin(filter_in)
    f_max = np.amax(filter_in)

    return (filter_in - f_min) * 1.0 / (f_max + 1e-5) * 255.0
