<div align="center">
  <img src="Figure/cat and dog.png" width="200">
</div>

<h1 align="center">
  Comprehensive Analysis of Heterogeneous Computing Performance of DNNs under typical frameworks on Cloud and Edge Computing Platforms
</h1>
 
<div align="center">
  <a href="https://pypi.org/project/pytorch/"><img src="https://img.shields.io/badge/pypi-v1.0.2-blue"></a>
  <a href="https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/install-guide/index.html"><img src="https://img.shields.io/badge/TensorRT-v7.1.3-blue"></a>
  <a href="https://www.ibm.com/products/spss-statistics"><img src="https://img.shields.io/badge/SPSS-v20-green"></a>
  <a href="https://www.originlab.com/index.aspx?go=PRODUCTS/Origin"><img src="https://img.shields.io/badge/Origin-v2019b-green"></a>
</div>
  
<div align="center">
  <a href="#dependencies"><b>🛠️Dependencies</b></a> |
  <a href="#introduce"><b>📘Introduce</b></a> |
  <a href="https://github.com/Foreverzfy/Model-Test/blob/master/Figure/Top-1.gif"><b>👀Dynamic Example</b></a>
</div>  
 
## Dependencies
* Python 3.7
* [PyTorch1.9.0](https://pytorch.org/) 
* Torchvision
* [Jetpack-4.4](https://developer.nvidia.com/jetpack-sdk-44-archive)
* [TensorRT 7.1.3](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/install-guide/index.html)
* [Google Colab](https://www.google.com/intl/zh-CN_ALL/drive/)
* [SPSS](https://www.ibm.com/analytics/spss-statistics-software)
* [OriginLab](https://www.originlab.com/)

## Usage

### Installation
Please refer to official website for the installation of [pytorch](https://pytorch.org/get-started/previous-versions/) and [dataset preparation](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

### Train and Test
Test the total parameters and G-FLOP value of each network model, run:
```
python flop.py
```

Use PyTorch framework to train and test, and run the following command at the terminal.
```
# train and test AlexNet
python AlexNet_test.py

# train and test other network models
python other_test.py
```

When using TensorRT framework to accelerate, you can run the following command to convert the trained pth file into an onnx file.
You can refer to the tutorial given by [NVIDIA official website](https://github.com/dusty-nv/jetson-inference) and run TensorRT to accelerate.
```
python pth_onnx.py
```

## Introduce
We selected nearly 40 commonly used deep neural network models, and analyzed their cross-platform and cross-inference frameworks in depth. By analyzing and comparing the obtained data, the performance of each model can be well grasped, and at the same time, some indexes of untested models can be predicted, which has guiding significance for future experiments and work. Some results are given below, and the specific results are shown in the paper.

### Accuracy and Model Complexity
We tested the relationship between the Top-1 accuracy of each model and the complexity of the model, drew the following bubble chart, and gave the drawing process.(The drawing process of all subsequent bubble charts is similar.)

<img src="Figure/1.png" width="400">

<h5>
  <img src="Figure/Top-1.gif" width="400">
</h5>
 
### Accuracy and Model Efficiency
Based on the obtained data, we drew the bubble chart of the relationship between the Top-1 accuracy and the accuracy density of each model.

<img src="Figure/2.png" width="400">

### Inference Time and Analysis Comparison
We get different Batch Size and inference time of different network models on Google Colab platform. See table below. See the paper for inference time on Jetson Nano.

<img src="Figure/3.png" width="400">

### Accuracy and Inference Time
We draw the bubble chart of the relationship between the Top-1 accuracy vs. images per second (FPS) of network models on the platforms of (a)Colab and, (b)Jetson Nano under the PyTorch inference framework, and (c) Jetson Nano under the TensorRT inference frameworks (batch size=1)

<img src="Figure/4.png" width="400">
(a)
<img src="Figure/5.png" width="400">
(b)
<img src="Figure/6.PNG" width="400">
(c)

### Memory Usage and Model Complexity
We draw a bubble chart of the relationship between GPU memory consumption and parameter content of different DNNs in Google Colab under Batch Size of 1.

<img src="Figure/7.png" width="400">

