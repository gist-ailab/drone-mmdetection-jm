import os
# dataset settings
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'FLIRCatDataset'
data_root = '/ailab_mat/dataset/FLIR_ADAS_v2'
backend_args = None

classes = ('person', 'bike', 'car', 'motor', 'bus', 'train', 'truck', 'light', 'hydrant','sign', 'dog', 'skaterboard', 'stroller',  'scooter', 'other Vehicle' )

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

model = dict(
    type = 'MultiModalAttFasterRCNN',
    data_preprocessor=dict(
        type='MultiModalDetDataPreprocessor',
    ),
    backbone=dict(
        in_channels=3
    ),
    neck=dict(
        in_channels=[
            256, 512, 1024, 2048,
        ],
        out_channels=128
    ),
    post_att =dict(
        type='SELayer',
        in_channels=256
    ),
    att = dict(
        type='SpatialATT'
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes)
        )
    ),
)


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadThermalImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RGBT_Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackMultiModalDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadThermalImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RGBT_Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackMultiModalDetInputs'),
]


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='CustomSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        metainfo=dict(classes=classes),
        type=dataset_type,
        data_root = os.path.join(data_root, 'video_rgb_test'),
        ann_file = 'train_coco_v4.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
        data_root=os.path.join(data_root,'video_rgb_test'),
        # ann_file='coco.json',
        ann_file='test_coco_v4.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root,'video_rgb_test','test_coco_v4.json'),
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator





