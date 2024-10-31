import json
import os
import os.path as osp
from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class FLIRCatDataset(CocoDataset):
    
    def __init__(self, *args, **kwargs):
        super(FLIRCatDataset, self).__init__(*args, **kwargs)

    def load_data_list(self):
        """Load RGB-Thermal paired annotations."""
        with open(self.ann_file) as f:
            coco_json = json.load(f)

        data_list = []
        for img_info in coco_json['images']:
            data = dict()
            data['img_path'] = osp.join(self.data_prefix['img'], img_info['file_name_RGB'])
            data['thermal_img_path'] = osp.join(self.data_prefix['thermal_img'], img_info['file_name_IR'])
            data['img_id'] = img_info['id']
            data['height'] = img_info['height']
            data['width'] = img_info['width']

            # Get corresponding annotations
            data['instances'] = [ann for ann in coco_json['annotations'] if ann['image_id'] == img_info['id']]
            data_list.append(data)

        return data_list

    def parse_data_info(self, raw_data_info: dict):
        """Parse data_info to include both RGB and Thermal paths."""
        data_info = {}
        data_info['img_path'] = raw_data_info['img_path']
        data_info['thermal_img_path'] = raw_data_info['thermal_img_path']
        data_info['img_id'] = raw_data_info['img_id']
        data_info['height'] = raw_data_info['height']
        data_info['width'] = raw_data_info['width']
        data_info['instances'] = raw_data_info['instances']
        return data_info
