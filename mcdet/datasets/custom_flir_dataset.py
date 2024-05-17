import json
import os.path as osp
from typing import List, Union
import os
import mmcv
from PIL import Image
from typing import List, Optional
from mmdet.datasets.coco import CocoDataset
import copy
import numpy as np

import mmengine
from mmdet.registry import DATASETS
import json


@DATASETS.register_module()
class FLIRCatDataset(CocoDataset):
    def __init__(self, *args, **kwargs):
        super(FLIRCatDataset, self).__init__(*args, **kwargs)
        self.rgb_to.thermal_map = json.load(open(os.path.join(self.data_root, 'rgb_to_thermal_vid_map.json')))

    def load_image(self, file_name, img_prefix):
        img_path = os.path.join(img_prefix, file_name)
        return mmcv.imread(img_path)

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        
        rgb_img = self.load_image(img_info['filename'], self.img_prefix)
        thermal_img_name = self.rgb_to_thermal_map[img_info['filename']]
        thermal_img = self.load_image(thermal_img_name, self.img_prefix.replace('video_rgb_test', 'video_thermal_test'))
        # Concatenate RGB and thermal images
        img = self.concat_images(rgb_img, thermal_img)
        results = dict(img=img, img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        rgb_img = self.load_image(img_info['filename'], self.img_prefix)
        thermal_img_name = self.rgb_to_thermal_map[img_info['filename']]
        thermal_img = self.load_image(thermal_img_name, self.img_prefix.replace('video_rgb_test', 'video_thermal_test'))
        
        # Concatenate RGB and thermal images
        img = self.concat_images(rgb_img, thermal_img)
        
        results = dict(img=img, img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def concat_images(self, rgb_img, thermal_img):
        return np.concatenate((rgb_img, thermal_img), axis=2)