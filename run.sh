#!/bin/bash

# Simple Sequential Training Script
echo "Starting sequential training..."

# 실험 1
echo "=== Experiment 1: StitchFusion CascadeRCNN ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch \
    --nproc_per_node=6 \
    --master_port=29500 \
    tools/train.py \
    custom_configs/DELIVER/hinton-deliver_stitchfusion_cascadercnn_lr0.01.py \
    --launcher pytorch

if [ $? -ne 0 ]; then
    echo "Experiment 1 failed!"
    exit 1
fi

echo "Experiment 1 completed. Waiting 30 seconds..."
sleep 30

# 실험 2
echo "=== Experiment 2: StitchFusion RCNN ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch \
    --nproc_per_node=6 \
    --master_port=29501 \
    tools/train.py \
    custom_configs/DELIVER/hinton-deliver_stitchfusion_rcnn_lr0.01.py \
    --launcher pytorch

if [ $? -ne 0 ]; then
    echo "Experiment 2 failed!"
    exit 1
fi

echo "Experiment 2 completed. Waiting 30 seconds..."
sleep 30

# 실험 3
echo "=== Experiment 3: CMNeXt DINO ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch \
    --nproc_per_node=6 \
    --master_port=29502 \
    tools/train.py \
    custom_configs/DELIVER/hinton-deliver-4scale_cmnext_dino.py \
    --launcher pytorch

if [ $? -ne 0 ]; then
    echo "Experiment 3 failed!"
    exit 1
fi

echo "Experiment 3 completed. Waiting 30 seconds..."
sleep 30

# 실험 4
echo "=== Experiment 4: StitchFusion DINO ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch \
    --nproc_per_node=6 \
    --master_port=29503 \
    tools/train.py \
    custom_configs/DELIVER/hinton-deliver-4scale_stitchfusion_dino.py \
    --launcher pytorch

if [ $? -ne 0 ]; then
    echo "Experiment 4 failed!"
    exit 1
fi

echo "All experiments completed successfully!"