# base config for custom FLIR ADAS v2
# drone-mmdetection-jm/custom_configs/Project_Drone/Paper/flir_rgbt/flir_adas.py
import os
_base_ = [
    '../../../../../configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../../../../../configs/_base_/datasets/coco_detection.py',
    '../../../../../configs/_base_/schedules/schedule_2x.py',
    '../../../../../configs/_base_/default_runtime.py'
]


dataset_type = 'FLIRRgbtCocoDataset'
backend_args = None
data_root = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/LLVIP_coco'
classes = ('bicycle', 'car', 'person', 'dog')

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadThermalImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RGBT_Resize', scale=(640, 512), keep_ratio=True),
    dict(type='PackMultiModalDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadThermalImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RGBT_Resize', scale=(640, 512), keep_ratio=True),
    dict(type='PackMultiModalDetInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='CustomSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        metainfo=dict(classes=classes),
        type=dataset_type,
        data_root = data_root,
        ann_file = 'annotations/train.json',
        # data_prefix=dict(img=''),
        data_prefix=dict(img='train_RGB'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='CustomSampler', shuffle=False),
    dataset=dict(
        metainfo=dict(classes=classes),
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        # data_prefix=dict(img=''),
        data_prefix=dict(visible='val_RGB'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader
val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root,'annotations','val.json'),
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator