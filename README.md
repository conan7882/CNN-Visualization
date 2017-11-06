# Visualization of Deep Covolutional Neural Networks


- This repository contains implementations of visualizatin of CNN in recent papers.
- The source code in the repository can be used to demostrate the algorithms as well as test on your own data.

## Requirements
- Python 3.3+
- [Tensorflow 1.3](https://www.tensorflow.org/)
- [TensorCV](https://github.com/conan7882/DeepVision-tensorflow) 


## Algorithms 

- [Gradient-weighted Class Activation Mapping (Grad-CAM)](https://github.com/conan7882/CNN-Visualization/tree/master/grad_cam) (ICCV 17)
- [Class Activation Mapping (CAM)](https://github.com/conan7882/CNN-Visualization/tree/master/class_activation_map) (CVPR 16)
- [Guided back propagation](https://github.com/conan7882/CNN-Visualization/tree/master/Guided_Backpropagation) (2014)

## Gradient-weighted Class Activation Mapping (Grad-CAM)
- Grad-CAM generates similar class heatmap as CAM, but it does not require to re-train the model for visualizatin.
- Details of the implementation and more results can be find [here](https://github.com/conan7882/CNN-Visualization/tree/master/grad_cam). Some results:

![grad-cam-result](grad_cam/figs/ex1.png)

## Class Activation Mapping (CAM)
- The class activation map highlights the most informative image regions relevant to the predicted class. This map can be obtained by adding a global average pooling layer at the end of convolutional layers.
- Details of the implementation and more results can be find [here](https://github.com/conan7882/CNN-Visualization/tree/master/class_activation_map). Some results:

![celtech_change](class_activation_map/figs/celtech_diff.png)


## Guided back propagation
<!--- Guided backpropagation generates clearer visulizations than deconvnet for higher layers.-->

- Details of the implementation and more results can be find [here](https://github.com/conan7882/CNN-Visualization/tree/master/Guided_Backpropagation). Some results:

![gbp](Guided_Backpropagation/figs/gbp.png)









