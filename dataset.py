#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import cv2
import numpy as np
import os
import random
import six


class LabeledImageDataset(chainer.datasets.LabeledImageDataset):

    def __init__(self, pairs, root, crop_size, mean=None, random=False,
                 dtype=np.float32, label_dtype=np.int32):

        super(LabeledImageDataset, self).__init__(
            pairs, root, dtype, label_dtype)
        self._crop_size = crop_size
        self._mean = mean
        self._random = random
        self._eigval = np.array([0.2175, 0.0188, 0.0045])
        self._eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                                 [-0.5808, -0.0045, -0.8140],
                                 [-0.5836, -0.6948, 0.4203]])

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, label = self._pairs[i]

        # load label data
        label = np.array(label, dtype=self._label_dtype)

        # load image
        full_path = os.path.join(self._root, path)
        image = cv2.imread(full_path)  # BGR order
        h, w = image.shape[:2]

        if self._random:
            # Resize image with its shorter side randomly sampled in [256, 480]
            if h < w:
                resize_h = random.randint(256, 480)
                resize_w = w * resize_h // h
            else:
                resize_w = random.randint(256, 480)
                resize_h = h * resize_w // w
            image = cv2.resize(image, (resize_w, resize_h))

            # Randomly crop a region
            top = random.randint(0, resize_h - self._crop_size)
            left = random.randint(0, resize_w - self._crop_size)

            # Randomly flip the image
            if random.randint(0, 1):
                image = image[:, ::-1, :]

        else:
            # Resize image with its shorter side = 256
            if h < w:
                resize_h = 256
                resize_w = w * resize_h // h
            else:
                resize_w = 256
                resize_h = h * resize_w // w
            image = cv2.resize(image, (resize_w, resize_h))

            # Crop the center
            top = (resize_h - self._crop_size) // 2
            left = (resize_w - self._crop_size) // 2

        # Crop a region
        bottom = top + self._crop_size
        right = left + self._crop_size
        image = image[top:bottom, left:right, :]

        # Substract mean and transpose
        image = image.transpose(2, 0, 1).astype(self._dtype)
        if self._mean is not None:
            image -= self._mean.astype(self._dtype)

        # Color augmentation (https://github.com/facebook/fb.resnet.torch
        # /blob/master/datasets/transforms.lua#L183)
        if self._random:
            alpha = np.random.normal(0, 0.1, 3)
            rgb = np.sum(self._eigvec * alpha * self._eigval, axis=1)
            for i in six.moves.range(3):
                image[i] += rgb[2 - i]

        return image, label
