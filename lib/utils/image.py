#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: image.py
# Author: Qian Ge <geqian1001@gmail.com>

from scipy import misc

def im_rescale(im, resize):
	im_shape = im.shape
	im = misc.imresize(im, (resize[0], resize[1], im_shape[-1]))
	return im