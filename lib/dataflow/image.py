#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: image.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np 
from tensorcv.dataflow.base import RNGDataFlow
from tensorcv.dataflow.normalization import identity


class DataFromFile(RNGDataFlow):
    """ Base class for image from files """
    def __init__(self,
                 ext_name,
                 data_dir='', 
                 num_channel=None,
                 shuffle=True,
                 normalize=None,
                 batch_dict_name=None,
                 normalize_fnc=identity):

        check_dir(data_dir)
        self.data_dir = data_dir
        self._shuffle = shuffle
        self._normalize = normalize
        self._normalize_fnc = normalize_fnc

        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        self.setup(epoch_val=0, batch_size=1)

        self._load_file_list(ext_name.lower())
        if self.size() == 0:
            print_warning('No {} files in folder {}'.\
                format(ext_name, data_dir))
        self.num_channels, self.im_size = self._get_im_size()
        self._data_id = 0

    def _load_file_list(self):
        raise NotImplementedError()

    def _suffle_file_list(self):
        pass

    def next_batch(self):
        assert self._batch_size <= self.size(), \
        "batch_size cannot be larger than data size"

        if self._data_id + self._batch_size > self.size():
            start = self._data_id
            end = self.size()
        else:
            start = self._data_id
            self._data_id += self._batch_size
            end = self._data_id
        # batch_file_range = range(start, end)
        batch_data = self._load_data(start, end)

        if end == self.size():
            self._epochs_completed += 1
            self._data_id = 0
            if self._shuffle:
                self._suffle_file_list()
        return batch_data

    def next_batch_dict(self):
        batch_data = self.next_batch()
        batch_dict = {name: data for name, data in zip(self._batch_dict_name, batch_data)}
        return batch_dict

    def _load_data(self, start, end):
        raise NotImplementedError()


class ImageFromFile(DataFromFile):
    def __init__(self,
                 ext_name,
                 data_dir='', 
                 num_channel=None,
                 shuffle=True,
                 normalize=None,
                 normalize_fnc=identity,
                 batch_dict_name=None,
                 pf=identity):
    
        if num_channel is not None:
            self.num_channels = num_channel
            self._read_channel = num_channel
        else:
            self._read_channel = None

        self._resize = get_shape2D(resize)
        self._resize_crop = resize_crop
        self._pf = pf

        super(ImageFromFile, self).__init__(ext_name, 
                                        data_dir=data_dir,
                                        shuffle=shuffle, 
                                        normalize=normalize,
                                        batch_dict_name=batch_dict_name,
                                        normalize_fnc=normalize_fnc)

    def _load_file_list(self, ext_name):
        im_dir = os.path.join(self.data_dir)
        self._im_list = get_file_list(im_dir, ext_name)
        if self._shuffle:
            self._suffle_file_list()

    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        self._im_list = self._im_list[idxs]

    def _load_data(self, start, end):
        input_im_list = []
        for k in range(start, end):
            im_path = self._im_list[k]
            im = load_image(im_path, read_channel=self._read_channel,
                            resize=self._resize,
                            resize_crop=self._resize_crop,
                            pf=self._pf)
            input_im_list.extend(im)

        # TODO to be modified 
        input_im_list = self._normalize_fnc(np.array(input_im_list), 
                                          self._get_max_in_val(), 
                                          self._get_half_in_val())
        return [input_im_list]

    def size(self):
        return self._im_list.shape[0]

    def get_data_list(self):
        return [self._im_list]

    def set_data_list(self, new_data_list):
        assert isinstance(new_data_list, list)
        assert len(new_data_list) == 1
        self._im_list = np.array(new_data_list[0])

    def set_pf(self, pf):
        self._pf = pf

    def suffle_data(self):
        self._suffle_file_list()
