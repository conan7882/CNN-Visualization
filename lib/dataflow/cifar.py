#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cifar.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import pickle

import numpy as np 

from tensorcv.dataflow.base import RNGDataFlow


## TODO Add batch size
class CIFAR(RNGDataFlow):
    def __init__(self, data_dir='', shuffle=True, batch_dict_name=None):
        self.num_channels = 3
        self.im_size = [32, 32]

        assert os.path.isdir(data_dir)
        self.data_dir = data_dir

        assert batch_dict_name is not None
        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        self.shuffle = shuffle

        self.setup(epoch_val=0, batch_size=1)
        self._file_list = [os.path.join(data_dir, 'data_batch_' + str(batch_id)) for batch_id in range(1,6)]

        # self._load_files()
        self._num_image = self.size()

        self._image_id = 0
        self._batch_file_id = -1
        self._image = []
        self._next_batch_file()

    def _next_batch_file(self):
        if self._batch_file_id >= len(self._file_list) - 1:
            self._batch_file_id = 0
            self._epochs_completed += 1
        else:
            self._batch_file_id += 1
        data_dict = unpickle(self._file_list[self._batch_file_id])
        self._image = np.array(data_dict['image'])
        self._label = np.array(data_dict['label'])

        if self.shuffle:
            self._suffle_files()

    def _suffle_files(self):
        idxs = np.arange(len(self._image))

        self.rng.shuffle(idxs)
        self._image = self._image[idxs]
        self._label = self._label[idxs]

    def size(self):
        try:
            return self.data_size
        except AttributeError:
            data_size = 0
            for k in range(len(self._file_list)):
                tmp_image = unpickle(self._file_list[k])['image']
                data_size += len(tmp_image)
            self.data_size = data_size
            return self.data_size
        
    def next_batch(self):
        # TODO assume batch_size smaller than images in one file
        assert self._batch_size <= self.size(), \
          "batch_size {} cannot be larger than data size {}".\
           format(self._batch_size, self.size())

        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_image = np.array(self._image[start:end])
        batch_label = np.array(self._label[start:end])

        if self._image_id + self._batch_size > len(self._image):
            self._next_batch_file()
            self._image_id = 0
            if self.shuffle:
                self._suffle_files()

        return batch_image, batch_label

    def next_batch_dict(self):
        re_dict = {}
        batch_data = self.next_batch()
        for key, data in zip(self._batch_dict_name, batch_data):
            re_dict[key] = data
        return re_dict


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    image = dict[b'data']
    labels = dict[b'labels']

    r = image[:,:32*32].reshape(-1,32,32)
    g = image[:,32*32: 2*32*32].reshape(-1,32,32)
    b = image[:,2*32*32:].reshape(-1,32,32)

    image = np.stack((r,g,b),axis=-1)

    return {'image': image, 'label': labels}

if __name__ == '__main__':
    a = CIFAR('D:\\Qian\\GitHub\\workspace\\tensorflow-DCGAN\\cifar-10-python.tar\\')
    t = a.next_batch()[0]
    print(t)
    print(t.shape)
    print(a.size())
    # print(a.next_batch()[0])
    # print(a.next_batch()[0])