
# /drone-mmdetection-jm/custom_configs/Project_Drone/Paper/llvip_rgbt/AttNet_r50_fpn_2x_llvip_rgbt_lr005.py
import os
_base_ = [
    './flir_adas.py'
]

data_root = '/SSDb/jemo_maeng/dset/data/FLIR_aligned_coco'

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001))

model = dict(
    type = 'MultiModalFasterRCNN',
    data_preprocessor=dict(
        type='MultiModalDetDataPreprocessor',
    ),
    backbone=dict(
        in_channels=6
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=4
        )
    )
)


train_dataloader = dict(
    dataset=dict(
        data_root = data_root,
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root = data_root,
    )
)

val_evaluator = dict(
    ann_file = os.path.join(data_root,'annotations','val.json'),
)
test_dataloader = val_dataloader
test_evaluator = val_evaluator