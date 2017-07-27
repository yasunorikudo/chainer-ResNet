#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import chainer
import cv2
import numpy as np
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--img', type=str, default='data/cat.png',
                        help='Path to image file.')
    parser.add_argument('--mean', type=str, default='data/mean.npy')
    parser.add_argument('--model', type=str, default='ResNet50',
                        choices=['ResNet50', 'ResNet101', 'ResNet152'])
    args = parser.parse_args()

    img = cv2.imread(args.img)
    img = cv2.resize(img, (224, 224))
    mean = np.load(args.mean)

    x_data = img.transpose(2, 0, 1).astype(np.float32)[np.newaxis, :, :, :]
    x_data -= mean

    model = __import__(args.model).ResNet()
    chainer.serializers.load_hdf5('{}.model'.format(args.model), model)
    model.train = False
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        model.to_gpu()
        x_data = chainer.cuda.to_gpu(x_data)

    x_data = chainer.Variable(x_data, volatile='on')
    pred = model(x_data, None)
    pred = chainer.cuda.to_cpu(pred.data)

    with open('data/synset_words.txt') as f:
        synset = f.read().split('\n')[:-1]

    for i in np.argsort(pred)[0][-1::-1][:5]:
        print(synset[i])
