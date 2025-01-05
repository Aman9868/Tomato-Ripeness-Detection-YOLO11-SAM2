# YOLOv11 & SAM2 Object Detection and Segmentation Tool

## Overview
Welcome to the **YOLOv11 & SAM2 Object Detection and Segmentation Tool**! This web-based application leverages the power of **YOLOv11** for object detection and **SAM2** for semantic segmentation to help you analyze images efficiently. Upload an image and witness real-time object detection and segmentation in action.

## Features
- **YOLOv11 Object Detection**: Detects objects in an image and draws bounding boxes around them.
- **SAM2 Segmentation**: Performs semantic segmentation based on detected objects.
- **Interactive UI**: Easy-to-use sliders to adjust confidence threshold, IoU threshold, and line width for YOLO detection.
- **Real-time Feedback**: See the results instantly with a responsive layout and progress bar.

![alt text](<Screenshot from 2025-01-05 17-33-19.png>)

## Requirements
Before using the tool, make sure to set up your environment with the necessary dependencies.

### Prerequisites:
- Python 3.x
- Install the following libraries:
  - `gradio`
  - `Pillow`
  - `ultralytics`
  - `python-dotenv`
  - `torch`
  - `torchvision`

You can install the required libraries with the following command:
```bash
pip install -r requirements.txt
```
