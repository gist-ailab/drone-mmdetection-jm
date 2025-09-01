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

data_root = '/ailab_mat2/dataset/drone/250312_sejong/drone_250312_sejong_multimodal_coco/'
dataset_type = 'SejongDetectionDataset'
classes = ('Enemy', 'LandingMarker', 'Obstacle', 'FireExt', 'Door', 'Victim', 'Ally', 'Exit', 'Window', 'Light')
backend_args = None

# 🔥 1. 데이터 파이프라인 재정의 (가장 중요한 변경점)
# -----------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(480, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(480, 640), keep_ratio=True),
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
    # rpn_head=dict(
    #     anchor_generator=dict(
    #         type='AnchorGenerator',
    #         scales=[8],  # 스케일은 기본값을 사용하거나 데이터셋 분석 후 수정 가능
    #         # 분석 결과로 얻은 새로운 비율을 여기에 적용합니다.
    #         ratios=[0.331, 0.650, 1.110, 2.048, 3.667],
    #         strides=[4, 8, 16, 32, 64])),
    
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,   # IoU가 0.5 이상이어야 Positive
                neg_iou_thr=0.5,
                min_pos_iou=0.5, # 모든 GT box에 대해 가장 IoU가 높은 proposal이 0.5 미만이라도 Positive로 할당
                match_low_quality=True, # True로 바꿔서 테스트해볼 수 있음
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    )
)

# DataLoader settings
train_dataloader = dict(
    batch_size=4,
    num_workers=4, # 🔥 워커 수 상향 조정
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f'{data_root}labels/train.json',
        data_prefix=dict(img='images'),
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
        ann_file=f'{data_root}labels/test.json',
        data_prefix=dict(img='images'),
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


experiment_name = 'sejong2504_faster_rcnn__v2'
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