#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import chainer
import numpy as np
import os
from progressbar import ProgressBar

from dataset import LabeledImageDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mean', type=str, default='data/mean.npy')
    parser.add_argument('--model', type=str, default='ResNet50',
                        choices=['ResNet50', 'ResNet101', 'ResNet152'])
    parser.add_argument('--root', type=str,
                        help='Path to validation images folder of ImageNet.')
    parser.add_argument('--val', type=str, default='data/val.txt')
    args = parser.parse_args()

    model = __import__(args.model).ResNet()
    chainer.serializers.load_hdf5('{}.model'.format(args.model), model)
    model.train = False
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        model.to_gpu()
        device = 0
    else:
        device = -1

    mean = np.load(args.mean)
    val = LabeledImageDataset(
        args.val, args.root, model.insize, mean, random=False)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.batchsize, repeat=False)

    sum_accuracy = 0
    count = 0
    pbar = ProgressBar(len(val))
    for batch in val_iter:
        x, t = chainer.dataset.concat_examples(batch, device=device)
        x = chainer.Variable(x, volatile='on')
        t = chainer.Variable(t, volatile='on')
        h = model(x, None)
        accuracy = float(chainer.functions.accuracy(h, t).data)
        sum_accuracy += accuracy * len(t)
        count += len(t)
        pbar.update(count)
    print('1-crop validation error (top-1) :', 1 - sum_accuracy / count)
