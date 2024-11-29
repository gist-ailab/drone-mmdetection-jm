#!/bin/bash

# Log all outputs to train_log.txt
# python tools/train_debug.py --config /media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/configs/custom/faster-rcnn_r50_fpn_2x_GISTindoor_splitFrame.py
python tools/train_debug.py --config /media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/configs/custom/faster-rcnn_r50_fpn_2x_GISTindoor_splitRandom_lr0.001.py
python tools/train_debug.py --config /media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/configs/custom/faster-rcnn_r18_fpn_2x_GISTindoor_splitFrame_lr0.001.py
# python tools/train_debug.py --config /media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/configs/custom/faster-rcnn_r101_fpn_2x_GISTindoor_splitFrame_lr0.0001.py
