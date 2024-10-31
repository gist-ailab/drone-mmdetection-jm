import json
import os.path as osp
from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class LLVIPDataset(CocoDataset):
    CLASSES = ('person',)  # Add more classes if available

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data_list(self):
        """Load paired visible and infrared annotations."""
        with open(self.ann_file) as f:
            coco_json = json.load(f)

        data_list = []
        for img_info in coco_json['images']:
            data = dict()
            # Assume visible and infrared images follow similar paths, replace as necessary
            data['visible_img_path'] = osp.join(self.data_prefix['visible'], img_info['file_name'])
            data['infrared_img_path'] = osp.join(self.data_prefix['infrared'], img_info['file_name'])
            data['img_id'] = img_info['id']
            data['height'] = img_info['height']
            data['width'] = img_info['width']
            # Get corresponding annotations for the image
            data['instances'] = [ann for ann in coco_json['annotations'] if ann['image_id'] == img_info['id']]
            data_list.append(data)

        return data_list

    def parse_data_info(self, raw_data_info: dict):
        """Parse and format data information."""
        data_info = {}
        data_info['visible_img_path'] = raw_data_info['visible_img_path']
        data_info['infrared_img_path'] = raw_data_info['infrared_img_path']
        data_info['img_id'] = raw_data_info['img_id']
        data_info['height'] = raw_data_info['height']
        data_info['width'] = raw_data_info['width']
        data_info['instances'] = raw_data_info['instances']

        return data_info
