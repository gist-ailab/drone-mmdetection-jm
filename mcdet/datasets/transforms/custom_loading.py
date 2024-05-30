# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import mmcv
import cv2
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile
from mmengine.fileio import get
from mmengine.structures import BaseDataElement

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks

from PIL import Image
import torchvision

@TRANSFORMS.register_module()
class LoadThermalImageFromFile(LoadImageFromFile):

    def transform(self, results: dict) -> dict:
        """Transform function to add thermal image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['thermal_img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = cv2.imread(results['thermal_img_path'])

        if self.to_float32:
            img = img.astype(np.float32)


        results['ir'] = img
        results['ir_img_shape'] = img.shape[:2]
        results['ir_ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class CatRGBT(LoadImageFromFile):
    def transform(self, results: dict) -> dict:
        """Transform function to add thermal image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['thermal_img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        # img = results['inputs']
        # thermal = results['thermal_inputs']
        # pil_thermal = topilimage(thermal)
        # pil_thermal = pil_thermal.resize((img.shape[2], img.shape[1]))
        # thermal = torchvision.transforms.ToTensor()(pil_thermal)
        # cat = torch.cat([img, thermal], dim=0) 
        # results['inputs'] = cat
        # return results

        pass