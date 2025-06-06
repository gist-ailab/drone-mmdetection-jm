# Default config for DELIVER detection dataset
# custom_configs/DELIVER/deliver_dataset.py
import os
_base_ = [
    '../../configs/_base_/schedules/schedule_2x.py',  # Fixed: correct relative path
    '../../configs/_base_/default_runtime.py'         # Fixed: correct relative path  
]

# Dataset basic info
dataset_type = 'DELIVERDetectionDataset'
data_root = '/mnt/nvme/workspace/AnySeg/data/DELIVER/'  # Added trailing slash
backend_args = None
classes = ('Vehicle', 'Human')

# Data preprocessor
data_preprocessor = dict(
    type='ListDataPreprocessor',
    mean_=[
        [0.485, 0.456, 0.406],       # RGB (ImageNet - same as DELIVER)
        [0.0, 0.0, 0.0],             # Depth  
        [0.0, 0.0, 0.0],             # Event
        [0.0, 0.0, 0.0]              # LiDAR
    ],
    std_=[
        [0.229, 0.224, 0.225],       # RGB (ImageNet - same as DELIVER)
        [1.0, 1.0, 1.0],             # Depth
        [1.0, 1.0, 1.0],             # Event
        [1.0, 1.0, 1.0]              # LiDAR
    ],
    pad_size_divisor=32
)

train_pipeline = [
    dict(type='LoadDELIVERImages'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='DELIVERResize', 
        img_scale=(512, 512), 
        keep_ratio=True,
        bbox_format='xywh'  # 🔥 xywh format 명시
    ),
    dict(
        type='DELIVERRandomFlip', 
        prob=0.5,
        bbox_format='xywh'  # 🔥 xywh format 명시
    ),
    dict(type='PackDELIVERDetInputs')
]

test_pipeline = [
    dict(type='LoadDELIVERImages'),
    dict(
        type='DELIVERResize', 
        img_scale=(512, 512), 
        keep_ratio=True,
        bbox_format='xywh'  # 🔥 xywh format 명시
    ),
    dict(type='PackDELIVERDetInputs')
]

# Dataset configs
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coco_train_xywh.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(
            classes=classes,
            palette=[(220, 20, 60), (119, 11, 32)]
        )
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coco_val_xywh.json',
        data_prefix=dict(img=''),  # Fixed: same as train
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(
            classes=classes,
            palette=[(220, 20, 60), (119, 11, 32)]
        )
    ),
)

test_dataloader = val_dataloader

# Evaluation settings  
val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'coco_val_xywh.json'),  # Fixed: consistent with dataset
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator

# Optimizer settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5, norm_type=2)
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,  # Updated for 2x schedule
        by_epoch=True,
        milestones=[16, 22],  # Updated for 2x schedule
        gamma=0.1
    )
]

# Training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)  # Updated for 2x
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# Visualization settings
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)

# Logging
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# Random seed
randomness = dict(seed=0, deterministic=False)