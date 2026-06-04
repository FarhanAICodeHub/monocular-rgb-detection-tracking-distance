# Monocular RGB-Based Object Detection, Tracking, and Distance Estimation

This repository contains the implementation and example results for a
monocular RGB-based road-scene perception framework. The framework
integrates YOLOv11 for object detection, BoT-SORT for multi-object
tracking, and a ResNet-18-based regression network for object distance
estimation.

## Overview

The system takes monocular RGB video frames as input and produces an
object-level perception output containing object class, bounding box,
tracking ID, confidence score, and estimated distance in meters.

## Overall Pipeline

![Overall Pipeline](figures/Overall_Framework%20%282%29.png)

## Dataset

This project uses the KITTI Tracking dataset. The dataset is not included
in this repository. Please download it from the official KITTI website.

Expected dataset structure:

```text
kitti_project/
└── training/
    ├── image_02/
    ├── label_02/
    └── calib/

## Pipeline

1. Convert KITTI Tracking annotations into YOLO format.
2. Train YOLOv11 for object detection.
3. Apply BoT-SORT for multi-object tracking.
4. Generate object-centric crops for tracked objects.
5. Train a ResNet-18 regression network for distance estimation.
6. Produce final output with class, ID, bounding box, confidence, and distance.

## Results

The trained YOLOv11 detector achieved an overall mAP@0.5 of 0.958.

The distance estimation module achieved an overall MAE of 1.01 m
with a standard deviation of 1.30 m.


## Notes

KITTI 3D object annotations are used only for distance supervision
during training. During inference, the system uses only monocular RGB
images.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt

Repository Structure
scripts/   Python scripts
figures/   Paper figures
results/   Experimental results
configs/   Configuration files
