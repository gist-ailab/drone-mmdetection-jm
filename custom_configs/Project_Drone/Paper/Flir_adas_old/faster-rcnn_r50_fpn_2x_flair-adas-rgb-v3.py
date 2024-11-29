import os
# dataset settings
_base_=[
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/flair-adas-paired.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/default_runtime.py'
]
dataset_type = 'CocoDataset'
data_root = '/ailab_mat/dataset/FLIR_ADAS_v2'
backend_args = None

classes = ('person', 'bike', 'car', 'motor', 'bus', 'train', 'truck', 'light', 'hydrant','sign', 'dog', 'skaterboard', 'stroller',  'scooter', 'other Vehicle' )

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
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
        data_root = os.path.join(data_root, 'images_rgb_train'),
        ann_file='coco.json',
        data_prefix=dict(img=''),
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
        data_root=os.path.join(data_root,'images_rgb_val'),
        ann_file='coco.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=os.path.join(data_root,'images_rgb_val','coco.json'),
#     metric='bbox',
#     format_only=False,
#     backend_args=backend_args)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root,'video_rgb_test','test_coco_v4.json'),
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes)
        )
    )
)