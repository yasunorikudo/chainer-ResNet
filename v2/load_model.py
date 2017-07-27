#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from chainer import serializers

import sys
sys.path.insert(0, './deep-residual-networks/caffe/python')
import caffe


def mk_list(res, k='a'):
    l = []
    for i in range(4 if k == 'a' else 3):
        l.append([res, k, 'conv{}'.format(i+1), 'W', 'data'])
        l.append([res, k, 'bn{}'.format(i+1), 'avg_mean'])
        l.append([res, k, 'bn{}'.format(i+1), 'avg_var'])
        l.append([res, k, 'bn{}'.format(i+1), 'gamma', 'data'])
        l.append([res, k, 'bn{}'.format(i+1), 'beta', 'data'])
    return l

def mk_res(data, block, layer):
    for line in mk_list('res{}'.format(block)):
        data.append(line)
    for i in range(layer-1):
        for line in mk_list('res{}'.format(block), 'b{}'.format(i+1)):
            data.append(line)
    return data

def get_model_num(name, R):
    r2, r3, r4, r5 = R
    if name[0] == 'res2': x = 3
    elif name[0] == 'res3': x = r2*9 + 6
    elif name[0] == 'res4': x = (r2+r3)*9 + 9
    elif name[0] == 'res5': x = (r2+r3+r4)*9 + 12

    if name[1] == 'a':
        if name[2] == 'conv4': x += 0
        elif name[2] == 'conv1': x += 3
        elif name[2] == 'conv2': x += 6
        elif name[2] == 'conv3': x += 9
        elif name[2] == 'bn4': x += 1
        elif name[2] == 'bn1': x += 4
        elif name[2] == 'bn2': x += 7
        elif name[2] == 'bn3': x += 10
    else:
        i = int(name[1].split('b')[1])
        x += i * 9 + 3
        if name[2] == 'conv1': x += 0
        elif name[2] == 'conv2': x += 3
        elif name[2] == 'conv3': x += 6
        elif name[2] == 'bn1': x += 1
        elif name[2] == 'bn2': x += 4
        elif name[2] == 'bn3': x += 7

    y = 0
    if name[3] == 'avg_var': y = 1
    elif name[3] == 'gamma': x += 1
    elif name[3] == 'beta':
        x += 1
        y = 1

    return x, y, (r2+r3+r4+r5)*9 + 15

def getattr_list(obj, name_list):
    for name in name_list:
        obj = getattr(obj, name)
    return obj

def setattr_part(obj, name_list, value, num):
    x = getattr_list(obj, name_list[:-num])
    setattr(x, name_list[len(name_list)-num], value)
    return x

def setattr_list(obj, name_list, value):
    x = setattr_part(obj, name_list, value, 1)
    for i in range(2, len(name_list)+1):
        x = setattr_part(obj, name_list, x, i)
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet152'])
    args = parser.parse_args()

    if args.model == 'resnet152':
        from resnet152 import ResNet
        R = (3, 8, 36, 3)
        layer = 152
    elif args.model == 'resnet101':
        from resnet101 import ResNet
        R = (3, 4, 23, 3)
        layer = 101
    elif args.model == 'resnet50':
        from resnet50 import ResNet
        R = (3, 4, 6, 3)
        layer = 50

    model = ResNet()
    net = caffe.Net('./data/ResNet-{}-deploy.prototxt'.format(layer),
                    './data/ResNet-{}-model.caffemodel'.format(layer), caffe.TEST)

    model.conv1.W.data   = net.params.items()[0][1][0].data
    model.bn1.avg_mean   = net.params.items()[1][1][0].data
    model.bn1.avg_var    = net.params.items()[1][1][1].data
    model.bn1.gamma.data = net.params.items()[2][1][0].data
    model.bn1.beta.data  = net.params.items()[2][1][1].data

    data = []
    for i, r in enumerate(R):
        data = mk_res(data, i + 2, r)
    for line in data:
        x, y, z = get_model_num(line, R)
        model = setattr_list(model, line, net.params.items()[x][1][y].data)

    model.fc.W.data = net.params.items()[z][1][0].data
    model.fc.b.data = net.params.items()[z][1][1].data

    serializers.save_npz('{}.npz'.format(args.model), model)
