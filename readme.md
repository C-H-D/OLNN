# Overview

Code for our otosclerosis diagnostic model.

# System Requirements

## Hardware Requirements

The package requires only a standard computer with GPU and enough RAM to support the operations defined by a user. 
For optimal performance, we recommend a computer with the following specs:

* RAM: 16+ GB
* CPU: 6+ cores, 2.8+ GHz/core
* GPU: 11+ GB (such as GeForce RTX 2080 Ti GPU)

## Software Requirements

This package is supported for Windows operating systems.

* Installing CUDA 10.0 on Windows 10
* Installing Python 3.6+ on Windows 10

Python Package Versions:

* numpy 1.17.3
* pytorch 1.3.1
* torchvision 0.4.2
* scipy 1.3.1
* scikit-image 0.16.2
* matplotlib 3.1.2
* pillow 6.2.1

# Installation Guide

A working version of CUDA, python and pytorch. This should be easy and simple installation. 

* CUDA(https://developer.nvidia.com/cuda-downloads)
* pytorch(https://pytorch.org/get-started/locally/) 
* python(https://www.python.org/downloads/)

# Usage of source code

## Training

* run create_dataset.py to create the training dataset. The data will be saved in "dataset/detect/" sub-folder.
* run train.py to train a network for detection and classification. The model will be save in "models" sub-folder.

## Test

run test.py to obtain initial results. The statistics will be shown directly.

## Evaluation

* run evaluate.py to obtain the results of evaluation including detection results and txt files specifying type and confidence of detection bounding box. The data will be saved in "dataset/eval_results" sub-folder.
* run result.py to obtain the final diagnostic result. This script should be run after evaluate.py and the results will be shown directly.