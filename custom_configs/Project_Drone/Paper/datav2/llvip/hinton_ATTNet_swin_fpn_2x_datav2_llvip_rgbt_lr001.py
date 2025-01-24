# base config for custom FLIR ADAS v2
# drone-mmdetection-jm/custom_configs/Project_Drone/Paper/flir_rgbt/flir_adas.py
import os
_base_ = [
    '../../../../../configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../../../../../configs/_base_/datasets/coco_detection.py',
    '../../../../../configs/_base_/schedules/schedule_2x.py',
    '../../../../../configs/_base_/default_runtime.py'
]


dataset_type = 'LLVIPRgbtDataset'
backend_args = None
data_root = '/SSDb/jemo_maeng/dset/data/DroneDataV2/LLVIP'
classes = ('person')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    type = 'MultiModalAttFasterRCNN',
    data_preprocessor=dict(
        type='MultiModalDetDataPreprocessor',
    ),

    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),

    neck = dict(
        in_channels=[96, 192, 384, 768],
        out_channels = 128
    ),
    post_att = dict(
        type = 'SELayer',
        in_channels = 256
    ),
    att = dict(
        type = 'SpatialATT'
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes)
        )
    ),
)


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadThermalImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RGBT_Resize', scale=(1024, 1280), keep_ratio=True),
    dict(type='PackMultiModalDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadThermalImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RGBT_Resize', scale=(1024, 1280), keep_ratio=True),
    dict(type='PackMultiModalDetInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        metainfo=dict(classes=classes),
        type=dataset_type,
        data_root = data_root,
        ann_file = 'coco_annotations/train.json',
        # data_prefix=dict(visible='train_RGB', infrared='train_thermal'),
        data_prefix =dict(visible ='visible/train', infrared='infrared/train'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=dict(classes=classes),
        type=dataset_type,
        data_root=data_root,
        ann_file='coco_annotations/val.json',
        data_prefix =dict(visible ='visible/test', infrared='infrared/test'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader
val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root,'coco_annotations','val.json'),
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator