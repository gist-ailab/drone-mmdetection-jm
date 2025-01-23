#%%
import os
import glob
import json
import numpy as np
from pycocotools.coco import COCO

# %%
from datetime import datetime
import glob
from collections import defaultdict
import json

class KAIST2COCO:
    def __init__(self, root, split):
        # self.category_map = {
        #     'person': 1,
        #     'people': 2,
        #     'cyclist': 3
        # }
        self.root = root
        self.ann_root = os.path.join(self.root, 'annotations')
        if split == 'train':
            self.split = os.path.join(self.root, 'splits', 'trainval.txt')
        elif split == 'test':
            self.split = os.path.join(self.root, 'splits', 'test.txt')
        self.coco_root = os.path.join(self.root, 'coco_annotations')
        os.makedirs(self.coco_root, exist_ok=True)
        self.img_id = 1

        self.coco_format={
                        "info": {
                "description": "KAIST Multispectral Pedestrian Detection Dataset",
                "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
                "version": "1.0",
                "year": 2015,
                "contributor": "KAIST",
                "date_created": datetime.now().strftime("%Y/%m/%d")
            },
            "licenses": [{
                "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
                "id": 1,
                "name": "KAIST Multispectral Pedestrian Dataset License"
            }],
            "images": [],
            "annotations": [],
            "categories": [
                {
                "supercategory": "person",
                "id": 1,
                "name": "person"
                },
                {
                "supercategory": "person",
                "id": 2,
                "name": "people"
                },
                {
                "supercategory": "person",
                "id": 3,
                "name": "cyclist"
                }
            ]
        }
    def parse_annotation(self, ann_file):
        annotations = []
        with open(ann_file, 'r') as f:
            lines =f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(' ')
                ann={
                    'category': parts[0],
                    'bbox' : [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])],
                }
                annotations.append(ann)
        return annotations

    def convert_dataset(self):
        indices = np.loadtxt(self.split, dtype=str)
        for idx in indices:
            sub_list = idx.split('/')
            ann_file_visible = glob.glob(os.path.join(self.ann_root,sub_list[0], sub_list[1], 'visible' , sub_list[2]+'*'))[0]
            ann_file_lwir = glob.glob(os.path.join(self.ann_root,sub_list[0], sub_list[1], 'lwir' , sub_list[2]+'*'))[0]
            ann_data = self.parse_annotation(ann_file_visible)
            image = {
                'file_name': os.path.join(sub_list[0], sub_list[1], 'visible', sub_list[2]+'.jpg'),
                'id': int(self.img_id),
                'width': 640,
                'height': 512
            }
            self.coco_format['images'].append(image)
            for ann in ann_data:
                category_name = ann['category']
                category_id = next((category['id'] for category in self.coco_format['categories'] if category['name'] == category_name), None)
                if category_id is None:
                    continue
                annotation = {
                    'image_id': int(self.img_id),
                    'category_id': int(category_id),
                    'bbox': ann['bbox'],
                    'id': len(self.coco_format['annotations'])+1,
                    'area': ann['bbox'][2]*ann['bbox'][3],
                    'iscrowd': 0
                }
                self.coco_format['annotations'].append(annotation)
            self.img_id+=1

    def save_coco(self):
        with open(os.path.join(self.coco_root, 'coco_{}.json'.format(self.split.split('/')[-1])), 'w') as f:
            json.dump(self.coco_format, f)

    def process(self):
        self.convert_dataset()
        self.save_coco()



def main():
    converter = KAIST2COCO('/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/SOTA/data_preparation/KAIST/dataset/kaist-paired', 'train')
    converter.process()


if __name__ == '__main__':
    main()