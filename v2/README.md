Deep Residual Network implementation by Chainer
========

Implementation by Chainer. Original paper is [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).

This repository includes network definition scripts, caffemodel converter and prediction scripts.

If you want to train ResNet from scratch, see [chainer sample code](https://github.com/pfnet/chainer/tree/master/examples/imagenet)

# Requirements

- [Chainer 2.0.1](https://github.com/pfnet/chainer) (Neural network framework)

- progressbar2 2.7.3

# Preparation

If you do not use Caffe, jump to "Download Chainer Model"

## Caffe SetUp

At chainer-ResNet root folder


Run:

```
git clone --recursive https://github.com/KaimingHe/deep-residual-networks.git
cd deep-residual-networks/caffe
```
After making Makefile.config

Run:
```
make -j16 && make pycaffe
```

## Convert Caffe Model to Chainer Model

Download Caffe models [here](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777).
Place Caffe models at $ROOT/data

Run:
```
python load_model.py --model ResNet152
```

## Download Chainer Model
If you do not want to use Caffe, you can download Chainer models.

Run:
```
wget https://www.dropbox.com/s/8snng201nxf1qw9/resnet50.npz
wget https://www.dropbox.com/s/srh8yo3h53od3tf/resnet101.npz
wget https://www.dropbox.com/s/344m52boylu13tf/resnet152.npz
```

# Predict

Run:

```
python predict.py --model resnet50 --img data/cat.png
```


# Evaluate pre-train model

For example, run:

```
python evaluate.py --model resnet50 --root /path/to/imagenet/val --gpu 0
```

1-crop validation error (top-1) for ImageNet validation set are as follows.

ResNet50  : 0.2479 (0.247)

ResNet101 : 0.2357 (0.236)

ResNet152 : 0.2319 (0.230)

The numbers in () are error rate that original repository reported. See [here](https://github.com/KaimingHe/deep-residual-networks).
