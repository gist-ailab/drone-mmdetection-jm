'''
Format:
python tools/test.py <config_file> <checkpoint_file> --show-dir <show_dir>
'''

python tools/test.py /media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/configs/custom/faster-rcnn_r101_fpn_2x_GISTindoor_spliFrameVideolr0.001.py \
/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/faster-rcnn_r101_fpn_2x_GISTindoor_spliFrameVideolr0.001/epoch_3.pth \
--show-dir viz_imgs/