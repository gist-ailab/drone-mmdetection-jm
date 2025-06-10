import os
_base_ = [
    './deliver_dataset.py'  # Inherit dataset config
]

# Base config의 optimizer 설정 제거
_delete_ = ['optim_wrapper']

data_root = '/SSDb/jemo_maeng/dset/DELIVER'

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
    type='DINO',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=data_preprocessor,  
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
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        num_classes=2,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

# train_pipeline - backend_args 문법 수정
train_pipeline = [
    dict(type='LoadDELIVERImages', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='DELIVERRandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='DELIVERRandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True,
                    bbox_format='xywh'
                )
            ],
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
                dict(
                    type='DELIVERRandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True,
                    bbox_format='xywh'
                )
            ]
        ]
    ),
    dict(type='PackDELIVERDetInputs')
]

# Validation pipeline 추가 (누락되어 있었음)
val_pipeline = [
    dict(type='LoadDELIVERImages', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='DELIVERRandomChoiceResize',
        scales=[(512, 512)],
        keep_ratio=True,
        bbox_format='xywh'
    ),
    dict(type='PackDELIVERDetInputs')
]

train_dataloader = dict(
    batch_size=1,  # 6 GPU * 2 = 12 total batch size
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
        pipeline=val_pipeline,  # 명시적으로 추가
    ),
)

test_dataloader = val_dataloader

# Evaluation settings  
val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'coco_val_xywh.json'),
    metric='bbox',
    format_only=False
)

# AdamW optimizer 설정 - base config의 SGD 설정을 완전히 덮어씀
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5, norm_type=2),
    accumulative_counts=16

)


# Learning policy
max_epochs = 200
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        T_max=max_epochs,
        eta_min=1e-6)
]
# Auto scale learning rate
auto_scale_lr = dict(enable=True, base_batch_size=16)  # 6 GPU * 2 batch_size

# Experiment name
experiment_name ='hinton-deliver-4scale_cmnext_dino_accu16_cosinlr'


# Override work_dir if needed
work_dir = f'./work_dirs/{experiment_name}'

# Environment settings for distributed training
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)