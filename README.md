# Lane Detection with UNet

![Inference Videos](https://github.com/AnshChoudhary/Lane-Detection-UNet/blob/main/header1.gif)

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
```
## Installation

1. Clone the repository:
```bash
git clone https://github.com/AnshChoudhary/Lane-Detection-UNet.git
cd Lane-Detection-UNet
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Training
The model is trained on NVIDIA A6000 GPU with 48GB VRAM. The training takes approximately 10-12 hours on these specs. To train the model, run:
```bash
CUDA_VISIBLE_DEVICES=<YOUR_GPU_ID> nohup python train.py
```

### Evaluation
Once the model is trained, you can evaluate the model's performance on the validation set (10,000 images) in termms of metrics like the Jaccard Score (IoU), Accuracy, and F1-Score. You can make necessary changes to eval_lane.py and then run the following command in order to evaluate the model:
```bash
CUDA_VISIBLE_DEVICES=<YOUR_GPU_ID> nohup python eval-lane.py
```

### Inference 
To run inference on a single image and save the predicted mask in the pred folder, use:
```bash
CUDA_VISIBLE_DEVICES=<YOUR_GPU_ID> nohup python inference.py
```

To run inference on a video and overlay the lane detection mask, use:
```bash
CUDA_VISIBLE_DEVICES=<YOUR_GPU_ID> nohup python video_infer2.py
```

To run inference on a video that would output an overlayed lane detection mask + YOLO detections, use:
```bash
CUDA_VISIBLE_DEVICES=<YOUR_GPU_ID> nohup python yolo_integrated.py
```

## Results
The model was evaluated on the following metrics over the validation set:
- Validation Jaccard Score (IoU): 0.9934
- Validation Accuracy: 0.9934
- Validation F1 Score: 0.9967

Here's a look at the model's predicted mask being compared to the ground truth mask on a sample image:

![Single Inference](https://github.com/AnshChoudhary/Lane-Detection-UNet/blob/main/Inference-PredMask.png)

Here's a look at a sample output video that overlays the lane detection mask from the trained model and performs YOLO object detections on cars, pedestrians, traffic lights, etc.:
