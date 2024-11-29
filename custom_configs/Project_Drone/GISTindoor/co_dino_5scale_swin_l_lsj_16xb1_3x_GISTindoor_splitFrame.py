import os

_base_ = ['../../../projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_1x_coco.py']


dataset_type = 'CocoDataset'
data_root = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/241108-indoor-gist/output'
backend_args = None

classes = ('human', 'Door', 'fire extinguisher', 'exit pannel', 'window')
num_classes = len(classes)

model = dict(backbone=dict(drop_path_rate=0.5),

             )


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
        ann_file = 'frame_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
        data_root= data_root,
        ann_file='frame_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root,'frame_val.json'),
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

param_scheduler = [dict(type='MultiStepLR', milestones=[30])]

train_cfg = dict(max_epochs=36)


val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root,'frame_val.json'),
    metric='bbox',
    format_only=False,
    backend_args=backend_args)