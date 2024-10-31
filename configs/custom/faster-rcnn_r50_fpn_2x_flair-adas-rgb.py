_base_=[
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/default_runtime.py'
]


image_size = (512, 640)

# Modify dataset configurations for training and validation
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='/ailab_mat/dataset/FLIR_ADAS_v2/images_rgb_train/coco.json',
        img_prefix='/ailab_mat/dataset/FLIR_ADAS_v2/images_rgb_train'),
    val=dict(
        type='CocoDataset',
        ann_file='/ailab_mat/dataset/FLIR_ADAS_v2/images_rgb_val/coco.json',
        img_prefix='/ailab_mat/dataset/FLIR_ADAS_v2/images_rgb_val'),
)

custom_imports = dict(
    imports=['mmdet_custom.hooks.validate_hook'], 
    allow_failed_imports=False
)

# Custom hook to save validation images
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='ValidationHook', interval=1000, save_path='/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/inferences/visualization/flir-rgb')
]

# Evaluation hook
evaluation = dict(interval=1000, metric='bbox', save_best='bbox_mAP_50')

# Logger settings to save outputs
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# Checkpoint hook to save the state of the model
checkpoint_config = dict(interval=1000)

# Set up to visualize the results after every 1000 iterations during training
custom_imports = dict(imports=['mmdet.core.visualization'], allow_failed_imports=False)
visualization = dict(
    type='DetVisualizationHook',
    interval=1000,
    out_dir='/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/inferences/visualization/flir-rgb',
)