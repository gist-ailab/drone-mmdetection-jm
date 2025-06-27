## how to visualization?

### 1. git clone my branch
### git clone -b gem git@github.com:gist-ailab/drone-mmdetection-jm.git
```cd drone-mmdetection-jm ```


### 2. python path 
```export PYTHONPATH=$(pwd)```


### 3. set your dataset path
### in custom_configs/DELIVER/deliver_dataset.py
data_root = '/SSDb/jemo_maeng/dset/DELIVER/'  <--- change your dataset path


### 4. model training
'''python tools/train.py custom_configs/DELIVER/hinton-deliver_geminifusion_rcnn_lr0.01.py'''


### 5. visualize the results
```python tools/test.py   custom_configs/DELIVER/hinton-deliver_geminifusion_rcnn_lr0.01.py   work_dirs/deliver_cmnext_b2_faster_rcnn_2x_lr0.01_2/best_coco_bbox_mAP_epoch_100.pth   --show-dir multi_modal_vis/```