# /media/jemo/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/mcdet/datasets/custom_deliver_detection_dataset.py

import json
import os
import os.path as osp
from typing import List, Optional
import mmcv
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS
import mmengine
import copy
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.datasets.api_wrappers import COCO
from typing import List, Union, Any
import torch

METAINFO = {
    'classes': ('Vehicle', 'Human'),  # DELIVER 클래스들
    'palette': [(220, 20, 60), (119, 11, 32)]  # 클래스별 색상
}

@DATASETS.register_module()
class DELIVERDetectionDataset(CocoDataset):
    """GIST dataset for object detection.
    Args:
        test_ratio (float): The ratio of the test set. Default: 0.1.
    """

    def __init__(self, *args, **kwargs):
        super(DELIVERDetectionDataset, self).__init__(*args, **kwargs)

        
    def full_init(self) -> None:
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - load_proposals: Load proposals from proposal file, if
              `self.proposal_file` is not None.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
            ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        self.data_list = self.load_data_list()
        print("[Loader info] datalist is loaded  : {}".format(len(self.data_list)))
        if self.proposal_file is not None:
            self.load_proposals()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        print('[Dataset info]: num_data = {}'.format(len(self.data_list)))

        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True


    def parse_data_info(self, raw_data_info: dict):
        """Parse raw annotation to target format."""
        
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None

        data_info['img_path'] = img_path
        data_info['depth_img_path'] = osp.join(self.data_prefix['img'], img_info['depth_path'])
        data_info['event_img_path'] = osp.join(self.data_prefix['img'], img_info['event_path'])
        data_info['lidar_img_path'] = osp.join(self.data_prefix['img'], img_info['lidar_path'])

        modality_paths = {
            'rgb': img_path,
            'depth': data_info['depth_img_path'],
            'event': data_info['event_img_path'], 
            'lidar': data_info['lidar_img_path']
        }    

        for modality, path in modality_paths.items():
            if not osp.exists(path):
                print(f"Warning: {modality} image not found: {path}")
        
        data_info['modality_paths'] = modality_paths
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        valid_instances = 0
        
        for i, ann in enumerate(ann_info):
            instance = {}
            if ann.get('ignore', False):
                continue

            x1, y1, w, h = ann['bbox']
            
            # 경계 검사
            x1 = max(0, x1)
            y1 = max(0, y1)
            w = min(w, img_info['width'] - x1)
            h = min(h, img_info['height'] - y1)
            
            # 기본적인 유효성 검사만 수행 (너무 엄격하지 않게)
            if w <= 1 or h <= 1:
                continue
                
            if ann['area'] <= 0:
                continue
                
            if ann['category_id'] not in self.cat_ids:
                continue

            bbox = [x1, y1, w, h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
                
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
            valid_instances += 1
        
        # 빈 샘플 처리
        if valid_instances == 0:
            print(f"Warning: No valid instances for {img_info['file_name']}, creating dummy instance")

        
        data_info['instances'] = instances
        return data_info
    
    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

