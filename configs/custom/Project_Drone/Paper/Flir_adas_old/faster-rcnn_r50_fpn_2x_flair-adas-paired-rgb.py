_base_=[
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/flair-adas-paired.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/default_runtime.py'
]


image_size = (512, 640)
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=80
        )
    )
)

# Modify dataset related settings



