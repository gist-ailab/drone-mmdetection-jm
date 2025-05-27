# CMNeXt Detection with RetinaNet detector
# custom_configs/DELIVER/deliver_cmnext_retinanet.py

_base_ = [
    './deliver_dataset.py'  # Inherit dataset config
]

# Model settings
model = dict(
    type='DELIVERRetinaNet',
    data_preprocessor=_base_.data_preprocessor,
    backbone=dict(
        type='CMNextBackbone',
        backbone='CMNeXt-B2',
        modals=['rgb', 'depth', 'event', 'lidar'],
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        pretrained='pretrained_weights/cmnext_b2_deliver.pth'
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=2,  # Vehicle, Human
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    # Training config
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1
        ),
        sampler=dict(
            type='PseudoSampler'  # RetinaNet uses all samples
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    # Testing config
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

# Experiment name
experiment_name = 'deliver_cmnext_b2_retinanet_2x'
work_dir = f'./work_dirs/{experiment_name}'