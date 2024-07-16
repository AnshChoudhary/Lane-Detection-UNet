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
- [Streamlit App](#streamlit-app)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Lane detection is a crucial component of autonomous driving systems. This project implements a UNet model to accurately segment lane markings from images. The UNet architecture is well-suited for this task due to its encoder-decoder structure that captures contextual information at multiple scales.

## Dataset
Our lane detection model is trained on the BDD100K dataset, which is ideal for this task due to:

- **Diversity**: It covers a wide range of driving scenarios, weather conditions, and times of day.
- **Rich Annotations**: It includes detailed annotations for lane markings, drivable areas, and objects.
- **Real-world Data**: Captured from real-world driving, ensuring the model generalizes well to actual driving conditions.
- **High Quality**: Provides high-resolution images necessary for accurate detection.
- **Community Support**: Widely used in the research community, providing benchmarks and continuous improvements.

By leveraging BDD100K, our model can perform lane detection effectively under various conditions, ensuring robust performance in all weather and lighting scenarios.
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

![Video Inference](https://github.com/AnshChoudhary/Lane-Detection-UNet/blob/main/output_input3_with_yolo_light.gif)

## Streamlit App
This project can be run on a streamlit web app in order to generate output videos that overlay the lane detection mask from the trained model and perform YOLO object detections. The user will be able to upload a video in avi, mov, mp4 formats and will have control over various parameterssuch as YOLO confidence threshold, detection transparency, interpolation factor, etc. 

Here's a look at the web app UI:

![Streamlit](https://github.com/AnshChoudhary/Lane-Detection-UNet/blob/main/lane-streamlit.png)

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/YourFeature`)
3. Commit your Changes (`git commit -m 'Add some YourFeature'`)
4. Push to the Branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/AnshChoudhary/Lane-Detection-UNet/blob/main/LICENSE) file for details.
