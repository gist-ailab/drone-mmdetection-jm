from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
import matplotlib.pyplot as plt
import mmcv
import cv2

config_file = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/configs/custom/faster-rcnn_r50_fpn_2x_flair-adas.py'
checkpoint_file= '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/faster-rcnn_r50_fpn_2x_flair-adas/epoch_24.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta
img = '/ailab_mat/dataset/FLIR_ADAS_v2/images_rgb_train/data/video-2BARff2EP7ZWkiF7n-frame-000477-5Wm39iF2QFc9LAcGh.jpg'
img = cv2.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')

result = inference_detector(model, img)
visualizer.add_datasample(
name='result',
image=img,
data_sample=result,
draw_gt=False,
pred_score_thr=0.3,
show=False)

img = visualizer.get_image()
img = mmcv.imconvert(img, 'bgr', 'rgb')
plt.imshow(img)
# cv2.imshow('result', img)

print("done")