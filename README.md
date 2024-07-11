# Lane Detection with UNet

![Inference Videos](https://github.com/AnshChoudhary/Lane-Detection-UNet/blob/main/4_videos%20(1).gif)

This repository contains code for training and evaluating a UNet model for lane detection using the BDD100K dataset. The project leverages PyTorch for model implementation and training, and includes scripts for preprocessing data, running inference, and evaluating model performance.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Lane detection is a crucial component of autonomous driving systems. This project implements a UNet model to accurately segment lane markings from images. The UNet architecture is well-suited for this task due to its encoder-decoder structure that captures contextual information at multiple scales.

## Dataset

The BDD100K dataset is used for training and evaluating the model. It includes diverse driving scenes with various weather conditions, times of day, and road types.

- Download the dataset from [BDD100K website](https://bdd-data.berkeley.edu/)

## Model Architecture

The UNet model is implemented with the following architecture:

- **Encoder**: A series of convolutional layers followed by batch normalization and ReLU activation.
- **Bottleneck**: A set of convolutional layers that capture the deepest features.
- **Decoder**: A series of transposed convolutional layers that upsample the features back to the original image size.

```python
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        # Define other layers...

    def forward(self, x):
        # Implement forward pass...
        pass
