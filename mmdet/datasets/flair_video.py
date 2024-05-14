import json
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset
from .coco import COCODataset
from typing import List, Optional
import copy

@DATASETS.register_module()
class FlairDataset(COCODataset):
    '''Dataset for FLIR-ADAS video dataset'''
    def __init__(self,
                 rgb_to_ir_map_file: str,
                 *args,
                 seg_map_suffix: str = '.png',
                 proposal_file: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 return_classes: bool = False,
                 caption_prompt: Optional[dict] = None,
                 **kwargs) -> None:
        self.seg_map_suffix = seg_map_suffix
        self.proposal_file = proposal_file
        self.backend_args = backend_args
        self.return_classes = return_classes
        self.caption_prompt = caption_prompt
        self.rgb_to_ir_map_file = rgb_to_ir_map_file
        self.rgb_to_ir_map = self.load_rgb_to_ir_map()
        self.thermal_prefix = self.image_prefix.replace('video_rgb_test', 'video_thermal_test')
        if self.caption_prompt is not None:
            assert self.return_classes, 'return_classes must be True when using caption_prompt'
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead'
            )
        super().__init__(*args, **kwargs)

    def load_rgb_to_ir_map(self):
        with open(self.rgb_to_ir_map_file, 'r') as file:
            return json.load(file)

    def load_data_list(self) -> List[dict]:
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = COCO(local_path)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(total_ann_ids), "Annotation ids are not unique!"
        del self.coco
        return data_list

    def __getitem__(self, index):
        data_info = self.data_list[index]
        img_id = data_info['raw_img_info']['img_id']
        filename = data_info['raw_img_info']['file_name']
        
        rgb_path = get_local_path(osp.join(self.data_root, filename), backend_args=self.backend_args)
        ir_path = get_local_path(osp.join(self.data_root, self.rgb_to_ir_map[filename]), backend_args=self.backend_args)

        rgb_img = self.img_prefix + filename
        ir_img = self.img_prefix + self.rgb_to_ir_map[filename]
        
        # Load images using mmengine's API or other preferred methods
        rgb_image = self.load_image(rgb_img)
        ir_image = self.load_image(ir_img)
        
        annotations = data_info['raw_ann_info']
        ann = self._parse_ann_info(annotations)
        
        if self.transforms:
            rgb_image, ir_image, ann = self.transforms(rgb_image, ir_image, ann)

        return rgb_image, ir_image, ann
