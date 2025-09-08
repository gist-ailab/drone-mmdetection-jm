# custom_configs/DELIVER/lecun-sejong2504_cmnext_rcnn_v3.py

import os
_base_ = [
    './deliver_dataset.py'
]

data_root = '/ailab_mat2/dataset/drone/250312_sejong/drone_250312_sejong_multimodal_coco/'
dataset_type = 'SejongDetectionDataset'
classes = ('Enemy', 'LandingMarker', 'Obstacle', 'FireExt', 'Door', 'Victim', 'Ally', 'Exit', 'Window', 'Light')

# Data pipelines (aspect ratio fixed)
train_pipeline = [
    dict(type='LoadDELIVERImages'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='DELIVERResize',
        img_scale=[(640, 480), (320, 240)], # Multi-scale (H, W)
        keep_ratio=True,
        bbox_format='xywh'
    ),
    dict(type='DELIVERRandomFlip', prob=0.5, bbox_format='xywh'),
    dict(type='PackDELIVERDetInputs')
]

test_pipeline = [
    dict(type='LoadDELIVERImages'),
    dict(type='DELIVERResize', img_scale=(640, 480), keep_ratio=True, bbox_format='xywh'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDELIVERDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Model settings
model = dict(
    type='FasterRCNN',
    # 🔥 For models with BatchNorm, SyncBN is recommended for multi-GPU training
    # If CMNextBackbone does not use BatchNorm, this can be omitted.
    # sync_bn=True,
    data_preprocessor=_base_.data_preprocessor,
    backbone=dict(
        type='CMNextBackbone',
        backbone='CMNeXt-B2',
        modals=['rgb', 'depth', 'event', 'lidar'],
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        pretrained='/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-mmdetection-jm/pretrained_weights/segformer/mit_b2.pth'
    ),
    # ... (Rest of the model definition is the same)
    neck=dict(type='FPN', in_channels=[64, 128, 320, 512], out_channels=256, num_outs=5),
    rpn_head=dict(
        type='RPNHead', in_channels=256, feat_channels=256,
        anchor_generator=dict(type='AnchorGenerator', scales=[2, 4, 8, 16], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[.0, .0, .0, .0], target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor', roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256, featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=10,
            bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.], target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1, pos_weight=-1, debug=False),
        rpn_proposal=dict(nms_pre=2000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1, debug=False)),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
)

# DataLoader settings
train_dataloader = dict(
    # 🔥 Batch size is PER GPU. Total batch size = 8 * (num_gpus)
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f'{data_root}labels/train.json',
        data_prefix=dict(img='images'),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes))
)

val_dataloader = dict(
    batch_size=1, # Validation is usually done on a single GPU per process
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
        metainfo=dict(classes=classes))
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'labels/test.json'),
    metric='bbox')

# Training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=50, by_epoch=True, begin=0, end=50, eta_min=1e-6)
]

# Optimizer (AdamW is generally a good starting point)
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
#     clip_grad=dict(max_norm=5, norm_type=2),
#     accumulative_counts=4
# )

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,  # 🔥 이 한 줄이 핵심! base의 optimizer 설정을 완전히 무시합니다.
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2))


# Default settings from _base_
default_hooks = _base_.default_hooks
log_processor = _base_.log_processor
vis_backends = _base_.vis_backends
visualizer = _base_.visualizer

# Experiment name
experiment_name = 'sejong2504_cmnext_b2_rcnn_multigpu_v3'
work_dir = f'./work_dirs/{experiment_name}'