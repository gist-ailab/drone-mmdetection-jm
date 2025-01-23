import os
# dataset settings
_base_=[
    '../../../../configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../../../../configs/_base_/datasets/flair-adas-paired.py',
    '../../../../configs/_base_/schedules/schedule_2x.py', 
    '../../../../configs/_base_/default_runtime.py'
]
dataset_type = 'CocoDataset'
# data_root = '/ailab_mat/dataset/FLIR_ADAS_v2/video_rgb_test'
data_root = '/SSDb/jemo_maeng/dset/data/FLIR_aligned_coco'

backend_args = None

# classes = ('person', 'bike', 'car', 'motor', 'bus', 'train', 'truck', 'light', 'hydrant','sign', 'dog', 'skaterboard', 'stroller',  'scooter', 'other Vehicle' )
classes = ('bicycle', 'car', 'person', 'dog')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
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
        # data_root = data_root+'/images_rgb_train',
        data_root = data_root,
        ann_file = 'annotations/train2.json',
        data_prefix=dict(img='train_RGB'),
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
        ann_file='annotations/val2.json',
        data_prefix=dict(img='val_RGB'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root,'annotations','val2.json'),
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes) +1
        )
    )
)