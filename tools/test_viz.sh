'''
Format:
python tools/test.py <config_file> <checkpoint_file> --show-dir <show_dir>
'''

python tools/test.py /SSDb/sangmin_park/drone-mmdetection-jm/custom_configs/DELIVER/hinton-deliver_geminifusion_rcnn_lr0.01.py  \
/SSDb/sangmin_park/drone-mmdetection-jm/work_dirs/deliver_cmnext_b2_faster_rcnn_2x_lr0.01_2/best_coco_bbox_mAP_epoch_100.pth \
--show-dir viz_imgs/