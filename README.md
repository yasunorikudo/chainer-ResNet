Deep Residual Network implementation by Chainer
========

Implementation by Chainer. Original paper is [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).

This repository includes network definition scripts, caffemodel converter and prediction scripts.

If you want to train ResNet from scratch, see [chainer sample code](https://github.com/pfnet/chainer/tree/master/examples/imagenet)

# Requirements

- [Chainer 1.5+](https://github.com/pfnet/chainer) (Neural network framework)


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
wget https://www.dropbox.com/s/4mvqcp9qr9ldjgc/ResNet50.model
wget https://www.dropbox.com/s/yqasroj1poru24u/ResNet101.model
wget https://www.dropbox.com/s/he2ndi7f9jixak6/ResNet152.model
```

# Predict

Run:

```
python predict.py --model ResNet152 --img cat.png
```
