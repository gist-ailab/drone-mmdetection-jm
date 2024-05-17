# Save this script as inference.py and run it

from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector, det_inferencer
import mmcv
import os
from tqdm import tqdm
import numpy as np
from viz_tools.custom_visualizer import vis_pred
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type = str, default = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/configs/custom/faster-rcnn_r50_fpn_2x_flair-adas-ir-v3.py', help= 'config file .py')
args = parser.parse_args()

# Load the configuration file
cfg = Config.fromfile(args.config)
# Set the checkpoint file
checkpoint_file = os.path.join('/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/', 'work_dirs', os.path.basename(args.config).replace('.py', ''), 'epoch_24.pth')
# checkpoint_file = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/faster-rcnn_r50_fpn_2x_flair-adas-rgb-v2/epoch_24.pth'
val_dir = os.path.join(cfg.data_root, 'images_rgb_val', 'data')
classes = cfg.classes
# List all images in the validation directory
val_images = [os.path.join(val_dir, img) for img in os.listdir(val_dir) if img.endswith('.jpg') or img.endswith('.png')]
# Directory to save the inference results
output_dir = os.path.join(os.path.dirname(checkpoint_file), 'inference_validation')
print(f"Saving results to {output_dir}")
os.makedirs(output_dir, exist_ok=True)
model = init_detector(cfg, checkpoint_file, device='cuda:0')
for img_path in tqdm(val_images):
    # Run inference
    result = inference_detector(model, img_path)
    # Load image
    img = mmcv.imread(img_path)
    img_result = vis_pred(img, result, classes)
    out_file = os.path.join(output_dir, os.path.basename(img_path))
    mmcv.imwrite(img_result, out_file)


