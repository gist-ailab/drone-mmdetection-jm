# CMNeXt Detection with RCNN detector
# custom_configs/DELIVER/deliver_cmnext_rcnn.py
import os
_base_ = [
    './deliver_dataset.py'  # Inherit dataset config
]

data_root= '/SSDb/jemo_maeng/dset/DELIVER'

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
            num_classes=2,  # Vehicle, Human
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

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        data_root=data_root,
        ann_file='coco_train_xywh.json',
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        ann_file='coco_val_xywh.json',
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
            name='hinton-deliver_cmnext_rcnn_lr0.01_ep50',
            tags=['CMNeXt', 'RCNN', 'full-finetune', 'epoch-50'],
            notes='CMNeXt RCNN with epoch 50 cosinelr',
            save_code=True
        ),
        define_metric_cfg=dict(
            loss_cls=dict(summary='min'),
            loss_bbox=dict(summary='min'),
            loss_iou=dict(summary='min'),
            loss=dict(summary='min'),
            bbox_mAP=dict(summary='max'),
            bbox_mAP_50=dict(summary='max'),
            bbox_mAP_75=dict(summary='max')
        )
    )
]


# Experiment name for logging
experiment_name = 'deliver_cmnext_b2_faster_rcnn_2x_cosinelr0.01_ep50'

# Override work_dir if needed
work_dir = f'./work_dirs/{experiment_name}'