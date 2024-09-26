# Save this script as inference.py and run it

from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector, det_inferencer
from mcdet import inference_rgbt_detector
import mmcv
import os
from tqdm import tqdm
import numpy as np
from viz_tools.custom_visualizer import vis_pred, vis_pred_w_gt
from tqdm import tqdm
from pycocotools.coco import COCO
import argparse
import json
import cv2
import os


class Inferer():
    def __init__(self, args):
        self.args = args
        self.cfg = Config.fromfile(args.config)
        self.checkpoint_file = os.path.join('/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/', 'work_dirs', os.path.basename(args.config).replace('.py', ''), 'epoch_24.pth')
        self.model = init_detector(self.cfg, self.checkpoint_file, device='cuda:0')
        self.val_dir = os.path.join(self.cfg.data_root, os.path.dirname(self.cfg.val_evaluator.ann_file).split('/')[-1])
        self.classes = self.cfg.classes
        self.output_dir = os.path.join(os.path.dirname(self.checkpoint_file), 'inference_pred_only')
        print(f"Saving results to {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True) 
        self.coco = COCO(self.cfg.val_evaluator.ann_file)
        self.img_ids = self.coco.getImgIds()
        #get class names from coco
        self.gt_classes = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.matching_json = json.load(open(self.args.rgbt_matching, 'r'))
        
    
    def run_inference(self):
        #Inference all validation image
        for i in tqdm(range(len(self.img_ids))):
            img_id = self.img_ids[i]  # Replace with your image id
            img_info = self.coco.loadImgs(img_id)[0]
            img_name = os.path.basename(img_info['file_name'])
            subfolder = os.path.dirname(img_info['file_name'])
            img_path = os.path.join(self.val_dir, img_info['file_name'])
            thermal_path = os.path.join(self.val_dir.replace('video_rgb_test', 'video_thermal_test'),subfolder, self.matching_json[img_name])
            result = inference_rgbt_detector(self.model, img_path, thermal_path)
            # Load image
            img = mmcv.imread(img_path)
            thermal = mmcv.imread(thermal_path)
            thermal = cv2.resize(thermal, (img.shape[1], img.shape[0]))
            # Write pred
            img_result = vis_pred(img, result, self.classes, score_thr=0.3)
            thermal_result = vis_pred(thermal, result, self.classes, score_thr=0.3)
            # Write GT
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            annotations = self.coco.loadAnns(ann_ids)
            # img_result = vis_pred_w_gt(img_result, result, annotations, self.classes, self.gt_classes, score_thr=0.5)
            # thermal_result = vis_pred_w_gt(thermal_result, result, annotations, self.classes, self.gt_classes, score_thr=0.5)
            out_file = os.path.join(self.output_dir, os.path.basename(img_path).replace('.jpg', '.png'))
            mmcv.imwrite(cv2.hconcat([img_result,thermal_result]), out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/configs/custom/faster-rcnn_r50_fpn_2x_flair-adas-rgbt-v3-preSpatialpostSE_LR0.005.py', help= 'config file *.py')
    parser.add_argument('--rgbt_matching', type = str, default = '/ailab_mat/dataset/FLIR_ADAS_v2/rgb_to_thermal_vid_map.json', help= 'rgb to thermal matching file')
    args = parser.parse_args()
    inferer = Inferer(args)
    inferer.run_inference()



if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES='5'
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    main()

