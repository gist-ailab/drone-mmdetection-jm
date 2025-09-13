# custom_configs/DELIVER/lecun-sejong2504_cmnext_rcnn_v2.py

import os
_base_ = [
    # './deliver_dataset.py',  # Inherit dataset config
    '../../configs/_base_/models/faster-rcnn_r50_fpn.py',
    # '../../configs/_base_/schedules/schedule_2x.py', 
    # '../../configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmdet.visualization'],
    allow_failed_imports=False)

data_root = '/ailab_mat2/dataset/drone/250312_sejong/drone_250312_sejong_multimodal_coco_cropped/'
dataset_type = 'CocoDataset'
classes = ('Enemy', 'LandingMarker', 'Obstacle', 'FireExt', 'Door', 'Victim', 'Ally', 'Exit', 'Window', 'Light')
backend_args = None

# -----------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
# -----------------------------------------------------------------
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes)
        )
    ),
)


# DataLoader settings
train_dataloader = dict(
    batch_size=32,
    num_workers=4, # 🔥 워커 수 상향 조정
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f'{data_root}labels/train_cropped.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=2),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes)
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
        ann_file=f'{data_root}labels/test_cropped.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'labels/test.json'),
    metric='bbox')
test_evaluator = val_evaluator

# Training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=50, # 🔥 전체 epoch 수와 일치
        by_epoch=True,
        begin=0, # 🔥 Warmup 직후부터 시작
        end=50,
        eta_min=1e-6)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5, norm_type=2),
    accumulative_counts=4
)


experiment_name = 'sejong2504_faster_rcnn_cropped_v2'
work_dir = f'./work_dirs/{experiment_name}'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='DELIVER',
            name=f'{experiment_name}',
            tags=['FasterRCNN', 'RCNN', 'full-finetune', 'epoch-50'],
            notes='FasterRCNN with epoch 50 SGD',
            save_code=True
        ),
    )
]

# ✅ Standard hooks configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook', 
        interval=50,
        log_metric_by_epoch=True,
        out_suffix='.log'
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=10,
        save_best='auto',
        max_keep_ckpts=3
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='DetVisualizationHook',
        draw=True,          # 시각화 비활성화 (성능 향상)
        interval=500,        # 간격 늘림
        show=False,
        wait_time=0.01
    ),
    step_tracker=dict(type='StepTrackerHook'),

    vis_log_resetter=dict(type='VisLogHook'),
)

# ✅ Simplified log processor
log_processor = dict(
    type='LogProcessor', 
    window_size=50, 
    by_epoch=True
)

# ✅ Visualizer 설정
visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)



# Experiment name
find_unused_parameters = True