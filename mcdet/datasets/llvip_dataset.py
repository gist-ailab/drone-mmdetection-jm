import json
import os.path as osp
from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS
import copy
from pycocotools.coco import COCO
@DATASETS.register_module()
class LLVIPRgbtDataset(CocoDataset):
    def __init__(self, *args, **kwargs):
        super(CocoDataset, self).__init__(*args, **kwargs)

    CLASSES = ('person',)  # Add more classes if available

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def load_data_list(self):
    #     """Load paired visible and infrared annotations."""
    #     with open(self.ann_file) as f:
    #         coco_json = json.load(f)
    #     self.coco = COCO(self.ann_file)
        
    #     categories = coco_json['categories']
    #     self.cat_ids = [category['id'] for category in categories]
    #     self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
    #     # self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

    #     data_list = []
    #     for img_info in coco_json['images']:
    #         data = dict()
    #         # Assume visible and infrared images follow similar paths, replace as necessary
    #         data['visible_img_path'] = osp.join(self.data_prefix['visible'], img_info['file_name'])
    #         # data['infrared_img_path'] = osp.join(self.data_prefix['infrared'], img_info['file_name'])
    #         data['infrared_img_path'] = osp.join(self.data_prefix['infrared'], img_info['file_name'])
    #         data['img_id'] = img_info['id']
    #         data['height'] = img_info['height']
    #         data['width'] = img_info['width']
    #         # Get corresponding annotations for the image
    #         data['instances'] = [ann for ann in coco_json['annotations'] if ann['image_id'] == img_info['id']]
    #         data_list.append(data)

    #     return data_list

    def parse_data_info(self, raw_data_info: dict):
        """Parse and format data information."""
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        img_name = img_info['file_name']
        data_info['img_path'] = osp.join(self.data_prefix['visible'], img_info['file_name'])
        # data_info['infrared_img_path'] =  osp.join(self.data_prefix['infrared'], img_info['file_name'])
        data_info['thermal_img_path'] =  osp.join(self.data_prefix['infrared'], img_info['file_name'])
        data_info['img_id'] = img_info['img_id']
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        # data_info['instances'] = img_info['instances']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True
            
        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = map(int, ann['bbox'])
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
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
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

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