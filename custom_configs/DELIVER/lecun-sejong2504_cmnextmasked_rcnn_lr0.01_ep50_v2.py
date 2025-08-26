# custom_configs/DELIVER/lecun-sejong2504_cmnext_rcnn_v2.py

import os
_base_ = [
    './deliver_dataset.py'  # Inherit dataset config
]

data_root = '/ailab_mat2/dataset/drone/250312_sejong/drone_250312_sejong_multimodal_coco/'
dataset_type = 'SejongDetectionDataset'
classes = ('Enemy', 'LandingMarker', 'Obstacle', 'FireExt', 'Door', 'Victim', 'Ally', 'Exit', 'Window', 'Light')

# 🔥 1. 데이터 파이프라인 재정의 (가장 중요한 변경점)
# -----------------------------------------------------------------
train_pipeline = [
    dict(type='LoadDELIVERImages'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='DELIVERResize',
        # 🔥 원본 비율(640x480)을 고려한 Multi-scale 학습. (H, W) 순서.
        #    모델이 다양한 크기의 객체를 학습하여 성능 향상에 도움이 됩니다.
        img_scale=[(480, 640),  (640, 852)], # 4:3 비율 유지
        keep_ratio=True,
        bbox_format='xywh'
    ),
    dict(
        type = 'DELIVERRandomMasking',
        masking_index=[0,1,2,3],
        mask_ratio = 0.3,
        patch_size=4,
    ),
    dict(
        type='DELIVERRandomFlip',
        prob=0.5,
        bbox_format='xywh'
    ),
    dict(type='PackDELIVERDetInputs')
]

# 검증/테스트 시에는 단일 스케일로 고정
test_pipeline = [
    dict(type='LoadDELIVERImages'),
    dict(
        type='DELIVERResize',
        img_scale=(480, 640), # 🔥 (H, W) 순서로 원본 비율 고정
        keep_ratio=True,
        bbox_format='xywh'
    ),
    dict(type='LoadAnnotations', with_bbox=True), # 🔥 GT 로드를 위해 추가
    dict(type='PackDELIVERDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
# -----------------------------------------------------------------


# Model settings (모델 구조는 변경 없음)
model = dict(
    type='FasterRCNN',
    data_preprocessor=_base_.data_preprocessor,
    backbone=dict(
        type='CMNeXBackbone',
        backbone='CMNeX-B2',
        modals=['rgb', 'depth', 'event', 'lidar'],
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        pretrained='/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-mmdetection-jm/pretrained_weights/segformer/mit_b2.pth'
        # pretrained='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/pretrained_weights/segformer/mit_b2.pth'
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))
    ),
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
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
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


# # Hooks, Logger, Visualizer (기존 설정 유지)
# default_hooks = _base_.default_hooks
# log_processor = _base_.log_processor
# vis_backends = _base_.vis_backends
# visualizer = _base_.visualizer

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='DELIVER',
            name='sejong2504_cmnextmasked_b2_rcnn_multiscale_v2',
            tags=['cmnext', 'RCNN', 'full-finetune', 'epoch-50'],
            notes='Stitfusion RCNN with epoch 50 SGD',
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
        draw=False,          # 시각화 비활성화 (성능 향상)
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
experiment_name = 'sejong2504_cmnextmasked_b2_rcnn_multiscale_v2'
work_dir = f'./work_dirs/{experiment_name}'