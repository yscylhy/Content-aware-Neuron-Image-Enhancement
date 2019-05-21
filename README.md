# Content-aware-Neuron-Image-Enhancement
This repo contains the implementation of our method on neuron image enhancement in ICIP 2016 [1] and TIP 2019 [2].

## Introduction
By exploring the property of sparsity and tube-like structure in the neuron images, we formulate high quality neuron images with a cost function. By minimizing the cost function iteratively, clutters and noise in neuron images are gradually removed. For more details about this work, please refer to our publications [1,2]. 


1. [Content-aware Neuron Image Enhancement](http://people.virginia.edu/~hl2uc/resources/papers/neuron_enhancement_v3.pdf), Haoyi Liang, Scott Acton and Daniel Weller, IEEE International Conf. on Image Processing, pp. 3510-3514, 2017	
2. [Content-Aware Enhancement of Images with Filamentous Structures](https://ieeexplore.ieee.org/document/8633852), Haris Jeelani, Haoyi Liang, Scott Acton and Daniel Weller, IEEE Trans. on Image Processing, 2019 

## Demonstration
### 2D example
![original image](images/2d_example.png "original image") ![enhanced image](assets/2d_example.png "enhanced image")

### 3D example
![enhanced image](assets/neuron_enhance.gif "3d enhancement example")

## Dependencies
1. `numpy`: matrix operation. Installation: `pip3 insall numpy`
2. `skimage`: Read and write image data. Installation: `pip3 install scikit-image`