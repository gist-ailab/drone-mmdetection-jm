# CMNeXt Detection with RCNN detector
# custom_configs/DELIVER/deliver_cmnext_rcnn.py
import os
_base_ = [
    './deliver_dataset.py'  # Inherit dataset config
]

data_root= '/SSDb/jemo_maeng/dset/DELIVER'

data_preprocessor = dict(
    type='ListDataPreprocessor',
    pad_mode='pad_to_max',  

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
# Model settings
model = dict(
    type='DeformableDETR',
    num_queries=300,
    num_feature_levels=4,
    with_box_refine=False,
    as_two_stage=False,
    data_preprocessor= data_preprocessor,  # This comes from _base_
    backbone=dict(
        type='CMNextBackbone',
        backbone='CMNeXt-B2',
        modals=['rgb', 'depth', 'event', 'lidar'],
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        pretrained='/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-mmdetection-jm/pretrained_weights/segformer/mit_b2.pth'
    ),

    neck=dict(
        type='ChannelMapper',
        in_channels=[64, 128, 320, 512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),

    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=2)  # Updated for 2x

train_pipeline = [
    dict(type='LoadDELIVERImages', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='DELIVERRandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            # [
            #     dict(
            #         type='DELIVERRandomChoiceResize',
            #         scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
            #                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
            #                 (736, 1333), (768, 1333), (800, 1333)],
            #         keep_ratio=True,
            #         bbox_format='xywh'
            #     )
            # ],
            [
                dict(
                    type='DELIVERRandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True,
                    bbox_format='xywh'
                ),
                dict(
                    type='DELIVERRandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                    bbox_format='xywh'
                ),
                # dict(
                #     type='DELIVERRandomChoiceResize',
                #     scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                #             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                #             (736, 1333), (768, 1333), (800, 1333)],
                #     keep_ratio=True,
                #     bbox_format='xywh'
                # )
            ]
        ]
    ),
    dict(type='PackDELIVERDetInputs')
]


train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        data_root=data_root,
        ann_file='coco_train_xywh.json',
        pipeline=train_pipeline,

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

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5, norm_type=2),
    accumulative_counts=8
)


# Experiment name for logging
experiment_name = os.path.splitext(os.path.basename(os.environ.get('CONFIG_FILE', 'default_config.py')))[0]

# Override work_dir if needed
work_dir = f'./work_dirs/{experiment_name}'