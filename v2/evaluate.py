#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import chainer
import multiprocessing
import numpy as np
import os
from progressbar import ProgressBar

from dataset import LabeledImageDataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mean', type=str, default='data/mean.npy')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--root', type=str,
                        help='Path to validation images folder of ImageNet.')
    parser.add_argument('--val', type=str, default='data/val.txt')
    args = parser.parse_args()
    if args.root is None or not os.path.exists(args.root):
        raise ValueError('Validation images folder \'{}\' does not exist.'.format(args.root))
    device = 0 if args.gpu >= 0 else -1

    # Load trained model.
    model = __import__(args.model).ResNet()
    chainer.serializers.load_npz('{}.npz'.format(args.model), model)
    model = chainer.links.Classifier(model)
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        model.to_gpu()

    # Create validation image iterator.
    mean = np.load(args.mean)
    val = LabeledImageDataset(args.val, args.root, model.predictor.insize, mean, random=False)
    multiprocessing.set_start_method('spawn')  # to avoid MultiprocessIterator's bug
    val_iter = chainer.iterators.MultiprocessIterator(val, args.batchsize, repeat=False)

    # Evaluation loop.
    chainer.config.enable_backprop = False
    chainer.config.train = False
    sum_accuracy = 0
    count = 0
    pbar = ProgressBar(len(val))
    for batch in val_iter:
        x, t = chainer.dataset.concat_examples(batch, device=device)
        x = chainer.Variable(x)
        t = chainer.Variable(t)
        loss = model(x, t)
        sum_accuracy += float(model.accuracy.data) * len(t)
        count += len(t)
        pbar.update(count)
    print('1-crop validation error (top-1) :', 1 - sum_accuracy / count)
    val_iter.finalize()
