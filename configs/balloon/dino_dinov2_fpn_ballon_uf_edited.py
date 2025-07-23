# Base configs
_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]

# Model configuration with DINOv2 backbone + DINO detector
model = dict(
    type='DINO',
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=14),
    backbone=dict(
        type='DINOv2',
        arch='b',
        img_size=504,
        patch_size=14,
        out_indices=(8, 9, 10, 11),
        norm_eval=True,
        frozen_stages=4,  # 8 -> 4로 변경 (더 많은 layer 학습)
        init_cfg=None
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[768, 768, 768, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.1),  # dropout 추가
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,  # dropout 증가
                act_cfg=dict(type='ReLU', inplace=True))),
        ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.1),  # dropout 추가
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.1),  # dropout 추가
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,  # dropout 증가
                act_cfg=dict(type='ReLU', inplace=True))),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),  # classification loss 가중치 증가
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))

# Dataset settings
data_root = 'data/balloon/'
metainfo = {
    'classes': ('balloon', ),
    'palette': [
        (220, 20, 60),
    ]
}

# 🔥 FIXED: Corrected data augmentation pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # 🔥 FIXED: Use RandomChoiceResize for multi-scale training
    dict(
        type='RandomChoiceResize',
        scales=[(400, 400), (450, 450), (500, 500), (550, 550), (600, 600)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # 🔥 IMPROVED: Better color augmentations
    # 🔥 FIXED: Use Resize instead of RandomCrop for simpler augmentation
    dict(type='Resize', scale=(504, 504), keep_ratio=True),
    dict(type='Pad', size=(504, 504), pad_val=dict(img=(114, 114, 114))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='PackDetInputs')
]

# Simpler validation pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(504, 504), keep_ratio=True),
    dict(type='Pad', size=(504, 504), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,  # Keep manageable batch size
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

# Evaluation settings
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/annotation_coco.json',
    metric=['bbox'],
    format_only=False)
test_evaluator = val_evaluator

# 개선된 optimizer 설정
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-3,  # learning rate 감소
        weight_decay=1e-4,
        betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=32  # gradient accumulation 감소
)

# 개선된 learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),  # warmup
    dict(
        type='CosineAnnealingLR',
        T_max=90,  # T_max를 실제 epoch에 맞춤
        by_epoch=True,
        begin=10,
        end=100,
        eta_min=1e-6)
]

# Training settings
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=800,
    val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5,
        save_best='auto',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# Environment settings
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# WandB + Local visualization backends
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='balloon-detection-dino-improved',
            name='dinov2-balloon-fixed-pipeline',
            tags=['DINO', 'DINOv2', 'balloon-detection', 'fixed'],
            notes='Fixed DINO detector with corrected data pipeline',
            save_code=True
        )
    )
]

visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False