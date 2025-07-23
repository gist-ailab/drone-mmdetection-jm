# Base configs
_base_ = [
    '../_base_/datasets/coco_detection.py',  # instance가 아닌 detection 사용
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
        pad_size_divisor=14),  # patch_size의 배수로 설정
    backbone=dict(
        type='DINOv2',
        arch='b',
        img_size=504,
        patch_size=14,
        out_indices=(8, 9, 10, 11),  # 마지막 4개 layer 사용
        norm_eval=True,
        frozen_stages=8,
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
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type='ReLU', inplace=True))),
        ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type='ReLU', inplace=True))),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=1,  # balloon 클래스 1개
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # denoising training
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    # training and testing settings
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

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),  # mask 제거
            dict(type='Resize', scale=(504, 504), keep_ratio=False),  # 고정 크기로 단순화
            dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]))

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
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(504, 504), keep_ratio=False),  # 고정 크기
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]))

test_dataloader = val_dataloader

# Evaluation settings
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/annotation_coco.json',
    metric=['bbox'],  # segm 제거
    format_only=False)
test_evaluator = val_evaluator

# Optimizer settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
        accumulative_counts=8
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        # milestones=[11],
        milestones=[70, 90],

        gamma=0.1)
]

# Training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# Environment settings
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='balloon-detection-dino',  # wandb 프로젝트 이름
            name='dinov2-balloon-exp',         # 실험 이름
            tags=['DINO', 'DINOv2', 'balloon-detection'],
            notes='DINO detector with DINOv2 backbone for balloon detection',
            save_code=True
        ),
        # 로그할 메트릭 정의 (선택사항)
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