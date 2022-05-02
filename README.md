# Classification-Algorithms-Pytorch
Algorithms for classification written in pytorch

## Introduction
I tried to implement algorithms used for classification using the pytorch library. I implemented the following algorithms

1. AlexNet (https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
![AlexNet-architecture-used-as-the-baseline-model-for-the-analysis-of-results-on-the](https://user-images.githubusercontent.com/45424924/137968374-5f74905f-9469-4cce-88e4-4c7f3c34e78a.png)

2. VGGNet (https://arxiv.org/abs/1409.1556)
![minerals-10-00958-g001](https://user-images.githubusercontent.com/45424924/137968570-9a8c82e5-381c-44dd-b9cb-352010c141d1.png)

3. EfficientNet (https://arxiv.org/abs/1905.11946)![1_vIZhPImFr9Gjpx6ZB7IOJg](https://user-images.githubusercontent.com/45424924/137969852-6e203b55-3551-478d-88b8-f0bdb810fa6a.png)

![The-EffecientNet-B0-general-architecture](https://user-images.githubusercontent.com/45424924/137968794-7c3ff4d0-767f-4a0c-9de5-675017e483b2.png)

4. InceptionNet (https://arxiv.org/abs/1409.4842)
![Inception-block-of-the-proposed-network-architecture-Here-n-stands-for-the-number-of](https://user-images.githubusercontent.com/45424924/137969121-1492f2b6-3f82-47e2-8694-90836c5d8977.png)

5. ResNet (https://arxiv.org/abs/1512.03385v1)
![The-representation-of-model-architecture-image-for-ResNet-152-VGG-19-and-two-layered](https://user-images.githubusercontent.com/45424924/137969235-e6fcd5e9-93f4-4dcf-91ca-6002486e49e1.png)

6. PreActResNet (https://arxiv.org/abs/1603.05027v3)

7. WideResNet (https://arxiv.org/abs/1605.07146v4)

8. ResNeXt (https://arxiv.org/abs/1611.05431v2)
![A-block-of-ResNet-Left-and-ResNeXt-with-cardinality-8-Right-A-layer-is-shown-as](https://user-images.githubusercontent.com/45424924/137970060-17a8fa0e-5515-4087-b7f1-1b5e13db67df.png)

9. DenseNet (https://arxiv.org/abs/1608.06993v4)
![O8ntGzS](https://user-images.githubusercontent.com/45424924/137970247-1d4c673f-ba4f-484e-8515-ca52b46602c6.png)


## How to use

## Requirements:software
Requirements for [PyTorch](http://pytorch.org/)

## Requirements:hardware
For most experiments, one or two K40(~11G of memory) gpus is enough cause PyTorch is very memory efficient. However,
to train DenseNet on cifar(10 or 100), you need at least 4 K40 gpus.

## Usage
1. Clone this repository

```
git clone https://github.com/Ti-Oluwanimi/Classification-Algorithms-Pytorch.git
```

2. Edit main.py and run.sh

In the ```main.py```, you can specify the network you want to train(for example):

```
model = resnet20_cifar(num_classes=10)

##Note
Please contact me if there are issues within the codebase. 
