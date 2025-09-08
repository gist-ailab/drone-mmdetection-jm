# CMNeXt Detection with RCNN detector
# custom_configs/DELIVER/deliver_cmnext_rcnn.py
import os
_base_ = [
    './deliver_dataset.py'  # Inherit dataset config
]

data_root= '/ailab_mat2/dataset/drone/250312_sejong/drone_250312_sejong_multimodal_coco/'
dataset_type = 'SejongDetectionDataset'
classes = ('Enemy', 'LandingMarker', 'Obstacle', 'FireExt', 'Door', 'Victim', 'Ally', 'Exit', 'Window', 'Light')
# Model settings
model = dict(
    type='FasterRCNN',
    data_preprocessor=_base_.data_preprocessor,  # This comes from _base_
    backbone=dict(
        type='CMNextBackbone',
        backbone='CMNeXt-B2',
        modals=['rgb', 'depth', 'event', 'lidar'],
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        pretrained='/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-mmdetection-jm/pretrained_weights/segformer/mit_b2.pth'
    ),
    neck=dict(
        type='FPN',  # MMDetection 표준 FPN 사용
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',  # 32 16 8 4 
            # scales=[8],
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
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,  # Vehicle, Human
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
    ),
    # Training config
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,        # 🔥 0.5 → 0.3
                neg_iou_thr=0.5,        # 🔥 0.5 → 0.1  
                min_pos_iou=0.5,        # 🔥 0.5 → 0.1
                match_low_quality=False, # 🔥 False → True
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),
    # Testing config
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
    )
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


train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type = dataset_type,
        data_root=data_root,
        ann_file=f'{data_root}labels/train.json',
        data_prefix=dict(img='images'),
        pipeline= train_pipeline,
        metainfo = dict(
            classes = classes,
            palette= [
            (220, 20, 60),     # Enemy - Crimson
            (0, 128, 0),       # LandingMarker - Green
            (0, 0, 255),       # Obstacle - Blue
            (255, 140, 0),     # FireExt - Dark Orange
            (255, 215, 0),     # Door - Gold
            (255, 0, 255),     # Victim - Magenta
            (0, 255, 255),     # Ally - Cyan
            (128, 0, 128),     # Exit - Purple
            (70, 130, 180),    # Window - Steel Blue
            (255, 255, 255)    # Light - White
            ]
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
        type = dataset_type,
        data_root=data_root,
        ann_file=f'{data_root}labels/test.json',
        data_prefix=dict(img='images'),
        test_mode = True,
        pipeline=test_pipeline,
        metainfo = dict(
            classes = classes,
            palette=[
            (220, 20, 60),     # Enemy - Crimson
            (0, 128, 0),       # LandingMarker - Green
            (0, 0, 255),       # Obstacle - Blue
            (255, 140, 0),     # FireExt - Dark Orange
            (255, 215, 0),     # Door - Gold
            (255, 0, 255),     # Victim - Magenta
            (0, 255, 255),     # Ally - Cyan
            (128, 0, 128),     # Exit - Purple
            (70, 130, 180),    # Window - Steel Blue
            (255, 255, 255)    # Light - White
            ]
        )
    ),
)

test_dataloader = val_dataloader

# Evaluation settings  
val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'labels/test.json'),  # Fixed: consistent with dataset
    metric='bbox',
    format_only=False
)
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=50, 
    val_interval=5)


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5, norm_type=2),
    accumulative_counts=4
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),  # warmup
    dict(
        type='CosineAnnealingLR',
        T_max=100,  # cosine annealing
        by_epoch=True,
        begin=10,
        end=50,
        eta_min=1e-6)
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='DELIVER',
            name='hinton-sejong2504_cmnext_rcnn_lr0.01_ep50',
            tags=['CMNeXt', 'RCNN', 'full-finetune', 'epoch-50'],
            notes='Custom sejong dataset, CMNeXt RCNN with epoch 50 cosinelr',
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
    )
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




# Experiment name for logging
experiment_name = 'sejong2504_cmnext_b2_faster_rcnn_2x_cosinelr0.01_ep50'

# Override work_dir if needed
work_dir = f'./work_dirs/{experiment_name}'