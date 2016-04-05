Deep Residual Network implementation by Chainer
========

Implementation by Chainer. Original paper is [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).

# Requirements

- [Chainer 1.5+](https://github.com/pfnet/chainer) (Neural network framework)


# Preparation

IF YOU DO NOT WANT TO USE CAFFE, JUMP TO "Download Chainer Model"

## Caffe SetUp

At chainer-ResNet root folder
run:

```
git clone --recursive https://github.com/KaimingHe/deep-residual-networks.git
cd deep-residual-networks/caffe  
```
After making Makefile.config
run:
```
make -j16 && make pycaffe
```

## Convert Caffe Model to Chainer Model

Download Caffe Models [here](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777).
Place Caffe Models at $ROOT/data
run:
```
python load_model.py --model ResNet152
```

## Download Chainer Model
If you do not want to use Caffe, you can download Chainer models.
run:
```
wget https://www.dropbox.com/s/4mvqcp9qr9ldjgc/ResNet50.model
wget https://www.dropbox.com/s/yqasroj1poru24u/ResNet101.model
wget https://www.dropbox.com/s/he2ndi7f9jixak6/ResNet152.model
```

# Predict

run:

```
python predict.py --model ResNet152 --img cat.png
```
