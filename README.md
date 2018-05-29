# Visualization of Deep Covolutional Neural Networks

[![Build Status](https://travis-ci.org/conan7882/CNN-Visualization.svg?branch=master)](https://travis-ci.org/conan7882/CNN-Visualization)
[![Coverage Status](https://coveralls.io/repos/github/conan7882/CNN-Visualization/badge.svg?branch=master)](https://coveralls.io/github/conan7882/CNN-Visualization?branch=master)

- This repository contains implementations of visualizatin of CNN in recent papers.
- The source code in the repository can be used to demostrate the algorithms as well as test on your own data.

## Requirements
- Python 3.3+
- [Tensorflow 1.3](https://www.tensorflow.org/)
- [TensorCV](https://github.com/conan7882/DeepVision-tensorflow) 


## Algorithms 

- [Visulization of filters and feature maps of GoogLeNet](https://github.com/conan7882/CNN-Visualization/tree/master/doc/firstfilter#visualization-of-filters-and-feature-maps-of-googlenet)
- [Gradient-weighted Class Activation Mapping (Grad-CAM)](https://github.com/conan7882/CNN-Visualization/tree/master/doc/grad_cam#gradient-weighted-class-activation-mapping-grad-cam) (ICCV'17)
- [Class Activation Mapping (CAM)](https://github.com/conan7882/CNN-Visualization/tree/master/doc/cam#class-activation-mapping-cam) (CVPR'16)
- [Guided back propagation](https://github.com/conan7882/CNN-Visualization/tree/master/doc/guided_backpropagation#guided-backpropagation) (2014)
- [Deconvolutional Networks](https://github.com/conan7882/CNN-Visualization/blob/master/doc/deconv/README.md#cnn-feature-visuallization-via-deconvnet-transposed-convolutional-layers) (ECCV'14)

## Visulization of filters and feature maps of GoogLeNet
- The most straightforward approach to visualize a CNN is to show the feature maps (activations) and filters.
- Details of the implementation and more results can be found [here](https://github.com/conan7882/CNN-Visualization/tree/master/doc/firstfilter#visualization-of-filters-and-feature-maps-of-googlenet)
<p align = 'left'>
<img src ="doc/firstfilter/figs/GoogLeNet.png" height="250" />
<img src ="doc/firstfilter/figs/GoogLeNet_inception3a.png" height="250" />
</p>

## Gradient-weighted Class Activation Mapping (Grad-CAM)
- Grad-CAM generates similar class heatmap as CAM, but it does not require to re-train the model for visualizatin.
- Details of the implementation and more results can be found [here](https://github.com/conan7882/CNN-Visualization/tree/master/doc/grad_cam#gradient-weighted-class-activation-mapping-grad-cam). Some results:

![grad-cam-result](doc/grad_cam/figs/ex1.png)

## Class Activation Mapping (CAM)
- The class activation map highlights the most informative image regions relevant to the predicted class. This map can be obtained by adding a global average pooling layer at the end of convolutional layers.
- Details of the implementation and more results can be found [here](https://github.com/conan7882/CNN-Visualization/tree/master/doc/cam#class-activation-mapping-cam). Some results:

![celtech_change](doc/cam/figs/celtech_diff.png)


## Guided back propagation
<!--- Guided backpropagation generates clearer visulizations than deconvnet for higher layers.-->

- Details of the implementation and more results can be found [here](https://github.com/conan7882/CNN-Visualization/tree/master/doc/guided_backpropagation#guided-backpropagation). Some results:

![gbp](doc/guided_backpropagation/figs/gbp.png)

## Deconvnet
- Pick a specific activation on a feature map and set other activation to zeros, then reconstruct an image by mapping back this new feature map to input pixel space.
- Details of the implementation and more results can be found [here](https://github.com/conan7882/CNN-Visualization/blob/master/doc/deconv/README.md#cnn-feature-visuallization-via-deconvnet-transposed-convolutional-layers). Some results:

<p align = 'left'>
<img src ="doc/deconv/figs/people/conv1_2_feat.png" height="200" />
<img src ="doc/deconv/figs/people/conv1_2_im.png" height="200" />
<img src ="doc/deconv/figs/people/conv2_2_feat.png" height="200" />
<img src ="doc/deconv/figs/people/conv2_2_im.png" height="200" />
<img src ="doc/deconv/figs/people/conv3_4_feat.png" height="200" />
<img src ="doc/deconv/figs/people/conv3_4_im.png" height="200" />
<img src ="doc/deconv/figs/people/conv4_4_feat.png" height="200" />
<img src ="doc/deconv/figs/people/conv4_4_im.png" height="200" />
</p>











