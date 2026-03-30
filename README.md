# UAV Drone Detection and Tracking - Assignment 3

## Overview
This project detects and tracks UAV drones in video using YOLOv8 and a Kalman filter.

## Task 1: Drone Object Detection

### Dataset
- **Source**: [dronenotdrone - Roboflow Universe](https://universe.roboflow.com/dronenotdrone/drone-detection-pq8sj)
- 7,010 images in YOLOv8 format
- Train: 4,914 | Valid: 1,396 | Test: 700

### Detector Configuration
- Model: YOLOv8s pretrained on COCO, fine-tuned on drone dataset
- Epochs: 3 | Image size: 300 | Batch size: 16
- Device: Apple MPS (M2 Pro)
- Confidence threshold: 0.15

### Detection Results
- 619 frames with drone detections saved across 2 test videos
- Detection frames dataset on Hugging Face: [Tom-Sorin/drone-detection-frames](https://huggingface.co/datasets/Tom-Sorin/drone-detection-frames)

### Failure Cases
- Small drones far from camera are missed due to limited training epochs
- Low mAP after few epochs means some drones are not detected
- Model may confuse birds or planes with drones at low confidence

## Task 2: Kalman Filter Tracking

### Kalman Filter Design
- State vector: [x, y, vx, vy] — 2D position and velocity of drone center
- Observation: [x, y] — detector bounding box center
- Library: filterpy
- Missing detections: tracker continues predicting for up to 10 consecutive missed frames

### Output Videos
- Video 1: https://youtu.be/KgqWaXK4bTk
- Video 2: https://youtu.be/hO6fs-lLZC8

### Failure Cases
- Tracker drifts when drone is occluded for more than 10 frames
- Fast-moving drones cause trajectory jumps due to high velocity noise

## Scripts
- `download_dataset.py` — downloads drone dataset from Roboflow
- `train.py` — fine-tunes YOLOv8s on the drone dataset
- `detect.py` — runs detection on all video frames and saves detections
- `track.py` — runs Kalman filter tracking and produces output videos
