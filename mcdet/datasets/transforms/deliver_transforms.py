# mcdet/datasets/transforms/deliver_transforms.py

import cv2
import mmcv
import numpy as np
import random
import torch
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import Resize, RandomCrop, RandomFlip
from typing import Dict, List, Tuple, Union, Optional
import copy

# Safe import for HorizontalBoxes
try:
    from mmdet.structures.bbox import HorizontalBoxes
except ImportError:
    try:
        from mmdet.structures import HorizontalBoxes
    except ImportError:
        # Fallback for older versions
        class HorizontalBoxes:
            def __init__(self, tensor):
                self.tensor = tensor


@TRANSFORMS.register_module()
class DELIVERResize:
    """Resize transform for DELIVER multimodal images with xywh bbox support - Fixed torch.clamp error."""
    
    def __init__(self, 
                 img_scale: Union[Tuple[int, int], List[Tuple[int, int]]] = None,
                 multiscale_mode: str = 'range',
                 ratio_range: Tuple[float, float] = None,
                 keep_ratio: bool = True,
                 bbox_clip_border: bool = True,
                 backend: str = 'cv2',
                 interpolation: str = 'bilinear',
                 bbox_format: str = 'xywh'):
        
        self.img_scale = img_scale
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.bbox_clip_border = bbox_clip_border
        self.backend = backend
        self.interpolation = interpolation
        self.bbox_format = bbox_format.lower()
    
    def _get_target_scale(self, ori_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Get target scale for resizing."""
        if isinstance(self.img_scale, list):
            if self.multiscale_mode == 'range':
                target_scale = random.choice(self.img_scale)
            else:
                target_scale = self.img_scale
        else:
            target_scale = self.img_scale
            
        if self.ratio_range is not None:
            ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
            target_scale = (int(target_scale[0] * ratio), int(target_scale[1] * ratio))
            
        return target_scale
    
    def _resize_img(self, img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """Resize single image."""
        return mmcv.imresize(img, new_shape, 
                           interpolation=self.interpolation, 
                           backend=self.backend)
    
    def _resize_bboxes_xywh(self, bboxes, scale_factor: Union[float, Tuple[float, float]], new_shape: Tuple[int, int]):
        """Resize bboxes in xywh format - Fixed torch.clamp error.
        
        Args:
            bboxes: bbox data in xywh format [x, y, width, height]
            scale_factor: float or (scale_x, scale_y) 
            new_shape: (height, width)
        """
        if bboxes is None:
            return bboxes
        
        # Normalize scale_factor to tuple format
        if isinstance(scale_factor, (int, float)):
            scale_x = scale_y = float(scale_factor)
        else:
            scale_x, scale_y = scale_factor[0], scale_factor[1]
            
        # Handle HorizontalBoxes object (MMDetection v3.x)
        if hasattr(bboxes, 'tensor'):
            if len(bboxes.tensor) == 0:
                return bboxes
                
            bbox_tensor = bboxes.tensor.clone()
            
            # xywh format: [x, y, w, h]
            bbox_tensor[:, 0] *= scale_x  # x
            bbox_tensor[:, 1] *= scale_y  # y  
            bbox_tensor[:, 2] *= scale_x  # width
            bbox_tensor[:, 3] *= scale_y  # height
            
            # 🔥 Fixed: Clip to image boundaries if needed
            if self.bbox_clip_border:
                # x >= 0, y >= 0 (Number clamp)
                bbox_tensor[:, 0] = torch.clamp(bbox_tensor[:, 0], 0, new_shape[1])
                bbox_tensor[:, 1] = torch.clamp(bbox_tensor[:, 1], 0, new_shape[0])
                
                # width, height <= remaining space (element-wise clamp)
                for i in range(bbox_tensor.shape[0]):
                    max_width = new_shape[1] - bbox_tensor[i, 0].item()
                    max_height = new_shape[0] - bbox_tensor[i, 1].item()
                    
                    bbox_tensor[i, 2] = torch.clamp(bbox_tensor[i, 2], 0, max_width)
                    bbox_tensor[i, 3] = torch.clamp(bbox_tensor[i, 3], 0, max_height)
            
            return HorizontalBoxes(bbox_tensor)
        
        # Handle numpy array
        elif isinstance(bboxes, np.ndarray):
            if len(bboxes) == 0:
                return bboxes
                
            bboxes = bboxes.copy()
            bboxes[:, 0] *= scale_x  # x
            bboxes[:, 1] *= scale_y  # y
            bboxes[:, 2] *= scale_x  # width  
            bboxes[:, 3] *= scale_y  # height
            
            if self.bbox_clip_border:
                bboxes[:, 0] = np.clip(bboxes[:, 0], 0, new_shape[1])
                bboxes[:, 1] = np.clip(bboxes[:, 1], 0, new_shape[0])
                
                # Element-wise clipping for width/height
                for i in range(bboxes.shape[0]):
                    max_width = new_shape[1] - bboxes[i, 0]
                    max_height = new_shape[0] - bboxes[i, 1]
                    bboxes[i, 2] = np.clip(bboxes[i, 2], 0, max_width)
                    bboxes[i, 3] = np.clip(bboxes[i, 3], 0, max_height)
                
            return bboxes
        
        # Handle list (instance format)
        elif isinstance(bboxes, list):
            resized_bboxes = []
            for bbox in bboxes:
                if isinstance(bbox, list):
                    bbox = np.array(bbox, dtype=np.float32)
                else:
                    bbox = bbox.copy()
                    
                bbox[0] *= scale_x  # x
                bbox[1] *= scale_y  # y
                bbox[2] *= scale_x  # width
                bbox[3] *= scale_y  # height
                
                if self.bbox_clip_border:
                    bbox[0] = np.clip(bbox[0], 0, new_shape[1])
                    bbox[1] = np.clip(bbox[1], 0, new_shape[0])
                    
                    max_width = new_shape[1] - bbox[0]
                    max_height = new_shape[0] - bbox[1]
                    bbox[2] = np.clip(bbox[2], 0, max_width)
                    bbox[3] = np.clip(bbox[3], 0, max_height)
                
                resized_bboxes.append(bbox.tolist())
            return resized_bboxes
        
        return bboxes
    
    def __call__(self, results: Dict) -> Dict:
        """Apply resize to all modalities."""
        if isinstance(results['img'], list):
            ori_shape = results['img'][0].shape[:2]  # (H, W)
            target_scale = self._get_target_scale(ori_shape)
            
            if self.keep_ratio:
                new_shape, scale_factor = mmcv.rescale_size(ori_shape, target_scale, return_scale=True)
            else:
                new_shape = target_scale
                scale_factor = (target_scale[1] / ori_shape[1], target_scale[0] / ori_shape[0])
            
            if isinstance(scale_factor, (int, float)):
                scale_x = scale_y = float(scale_factor)
            else:
                scale_x, scale_y = scale_factor[0], scale_factor[1]
            
            # Resize all modality images
            resized_imgs = []
            for img in results['img']:
                resized_img = self._resize_img(img, new_shape)
                resized_imgs.append(resized_img)
            
            results['img'] = resized_imgs
            results['img_shape'] = resized_imgs[0].shape[:2]
            results['scale_factor'] = (scale_x, scale_y)
            
            # 🔥 bbox format에 따라 다른 함수 사용
            if 'gt_bboxes' in results:
                if self.bbox_format == 'xywh':
                    results['gt_bboxes'] = self._resize_bboxes_xywh(results['gt_bboxes'], (scale_x, scale_y), new_shape)
                # else: xyxy format handling (if needed)
            
            # Update instance bboxes (xywh format)
            if 'instances' in results:
                for instance in results['instances']:
                    if 'bbox' in instance:
                        bbox = np.array(instance['bbox'], dtype=np.float32)
                        
                        if self.bbox_format == 'xywh':
                            # xywh format
                            bbox[0] *= scale_x  # x
                            bbox[1] *= scale_y  # y
                            bbox[2] *= scale_x  # width
                            bbox[3] *= scale_y  # height
                            
                            if self.bbox_clip_border:
                                bbox[0] = np.clip(bbox[0], 0, new_shape[1])
                                bbox[1] = np.clip(bbox[1], 0, new_shape[0])
                                
                                max_width = new_shape[1] - bbox[0]
                                max_height = new_shape[0] - bbox[1]
                                bbox[2] = np.clip(bbox[2], 0, max_width)
                                bbox[3] = np.clip(bbox[3], 0, max_height)
                        
                        instance['bbox'] = bbox.tolist()
        
        return results


@TRANSFORMS.register_module()
class DELIVERRandomCrop:
    """Random crop transform for DELIVER multimodal images with xywh bbox support - Fixed torch.clamp error."""
    
    def __init__(self, 
                 crop_size: Tuple[int, int],
                 crop_type: str = 'absolute',
                 allow_negative_crop: bool = False,
                 bbox_clip_border: bool = True,
                 bbox_format: str = 'xywh'):
        
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.bbox_format = bbox_format.lower()
    
    def _get_crop_bbox(self, img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get random crop bbox."""
        h, w = img_shape[:2]
        crop_h, crop_w = self.crop_size
        
        if crop_h >= h and crop_w >= w:
            return 0, 0, w, h
        
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        return x1, y1, x2, y2
    
    def _crop_img(self, img: np.ndarray, crop_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop single image."""
        x1, y1, x2, y2 = crop_bbox
        return img[y1:y2, x1:x2]
    
    def _crop_bboxes_xywh(self, bboxes, crop_bbox: Tuple[int, int, int, int]):
        """Crop bboxes in xywh format - Fixed torch.clamp error."""
        if bboxes is None:
            return bboxes
            
        x1, y1, x2, y2 = crop_bbox
        crop_w, crop_h = x2 - x1, y2 - y1
        
        # Handle HorizontalBoxes object
        if hasattr(bboxes, 'tensor'):
            if len(bboxes.tensor) == 0:
                return bboxes
                
            bbox_tensor = bboxes.tensor.clone()
            
            # xywh format: [x, y, w, h]
            bbox_tensor[:, 0] -= x1  # x coordinate
            bbox_tensor[:, 1] -= y1  # y coordinate
            
            if self.bbox_clip_border:
                # Clip x, y to be within crop region (Number clamp)
                bbox_tensor[:, 0] = torch.clamp(bbox_tensor[:, 0], 0, crop_w)
                bbox_tensor[:, 1] = torch.clamp(bbox_tensor[:, 1], 0, crop_h)
                
                # Element-wise clipping for width/height
                for i in range(bbox_tensor.shape[0]):
                    max_width = crop_w - bbox_tensor[i, 0].item()
                    max_height = crop_h - bbox_tensor[i, 1].item()
                    bbox_tensor[i, 2] = torch.clamp(bbox_tensor[i, 2], 0, max_width)
                    bbox_tensor[i, 3] = torch.clamp(bbox_tensor[i, 3], 0, max_height)
                
            return HorizontalBoxes(bbox_tensor)
        
        # Handle numpy array
        elif isinstance(bboxes, np.ndarray):
            if len(bboxes) == 0:
                return bboxes
                
            bboxes = bboxes.copy()
            bboxes[:, 0] -= x1  # x coordinate
            bboxes[:, 1] -= y1  # y coordinate
            
            if self.bbox_clip_border:
                bboxes[:, 0] = np.clip(bboxes[:, 0], 0, crop_w)
                bboxes[:, 1] = np.clip(bboxes[:, 1], 0, crop_h)
                
                # Element-wise clipping
                for i in range(bboxes.shape[0]):
                    max_width = crop_w - bboxes[i, 0]
                    max_height = crop_h - bboxes[i, 1]
                    bboxes[i, 2] = np.clip(bboxes[i, 2], 0, max_width)
                    bboxes[i, 3] = np.clip(bboxes[i, 3], 0, max_height)
                
            return bboxes
        
        return bboxes
    
    def __call__(self, results: Dict) -> Dict:
        """Apply crop to all modalities."""
        if isinstance(results['img'], list):
            img_shape = results['img'][0].shape
            crop_bbox = self._get_crop_bbox(img_shape)
            
            # Crop all modality images
            cropped_imgs = []
            for img in results['img']:
                cropped_img = self._crop_img(img, crop_bbox)
                cropped_imgs.append(cropped_img)
            
            results['img'] = cropped_imgs
            results['img_shape'] = cropped_imgs[0].shape[:2]
            
            # Crop bboxes
            if 'gt_bboxes' in results:
                if self.bbox_format == 'xywh':
                    results['gt_bboxes'] = self._crop_bboxes_xywh(results['gt_bboxes'], crop_bbox)
            
            # Update instance bboxes (xywh format)
            if 'instances' in results:
                x1, y1, x2, y2 = crop_bbox
                crop_w, crop_h = x2 - x1, y2 - y1
                
                for instance in results['instances']:
                    if 'bbox' in instance:
                        bbox = np.array(instance['bbox'], dtype=np.float32)
                        
                        if self.bbox_format == 'xywh':
                            bbox[0] -= x1  # x coordinate
                            bbox[1] -= y1  # y coordinate
                            
                            if self.bbox_clip_border:
                                bbox[0] = np.clip(bbox[0], 0, crop_w)
                                bbox[1] = np.clip(bbox[1], 0, crop_h)
                                
                                max_width = crop_w - bbox[0]
                                max_height = crop_h - bbox[1]
                                bbox[2] = np.clip(bbox[2], 0, max_width)
                                bbox[3] = np.clip(bbox[3], 0, max_height)
                        
                        instance['bbox'] = bbox.tolist()
        
        return results


@TRANSFORMS.register_module()
class DELIVERRandomFlip:
    """Random flip transform for DELIVER multimodal images with xywh bbox support."""
    
    def __init__(self, 
                 prob: float = 0.5,
                 direction: str = 'horizontal',
                 bbox_format: str = 'xywh'):
        
        self.prob = prob
        self.direction = direction
        self.bbox_format = bbox_format.lower()
    
    def _flip_img(self, img: np.ndarray, direction: str) -> np.ndarray:
        """Flip single image."""
        return mmcv.imflip(img, direction=direction)
    
    def _flip_bboxes_xywh(self, bboxes, img_shape: Tuple[int, int], direction: str):
        """Flip bboxes in xywh format."""
        if bboxes is None:
            return bboxes
            
        h, w = img_shape[:2]
        
        # Handle HorizontalBoxes object
        if hasattr(bboxes, 'tensor'):
            if len(bboxes.tensor) == 0:
                return bboxes
                
            bbox_tensor = bboxes.tensor.clone()
            
            if direction == 'horizontal':
                # xywh format: x_new = img_width - (x_old + width)
                bbox_tensor[:, 0] = w - (bbox_tensor[:, 0] + bbox_tensor[:, 2])
                
            elif direction == 'vertical':
                # y_new = img_height - (y_old + height)
                bbox_tensor[:, 1] = h - (bbox_tensor[:, 1] + bbox_tensor[:, 3])
                
            return HorizontalBoxes(bbox_tensor)
        
        # Handle numpy array
        elif isinstance(bboxes, np.ndarray):
            if len(bboxes) == 0:
                return bboxes
                
            bboxes = bboxes.copy()
            
            if direction == 'horizontal':
                bboxes[:, 0] = w - (bboxes[:, 0] + bboxes[:, 2])
            elif direction == 'vertical':
                bboxes[:, 1] = h - (bboxes[:, 1] + bboxes[:, 3])
                
            return bboxes
        
        return bboxes
    
    def __call__(self, results: Dict) -> Dict:
        """Apply flip to all modalities."""
        if isinstance(results['img'], list):
            flip = random.random() < self.prob
            
            if flip:
                img_shape = results['img'][0].shape
                
                # Flip all modality images
                flipped_imgs = []
                for img in results['img']:
                    flipped_img = self._flip_img(img, self.direction)
                    flipped_imgs.append(flipped_img)
                
                results['img'] = flipped_imgs
                
                # Flip bboxes
                if 'gt_bboxes' in results:
                    if self.bbox_format == 'xywh':
                        results['gt_bboxes'] = self._flip_bboxes_xywh(results['gt_bboxes'], img_shape, self.direction)
                
                # Update instance bboxes (xywh format)
                if 'instances' in results:
                    h, w = img_shape[:2]
                    
                    for instance in results['instances']:
                        if 'bbox' in instance:
                            bbox = np.array(instance['bbox'], dtype=np.float32)
                            
                            if self.bbox_format == 'xywh':
                                if self.direction == 'horizontal':
                                    bbox[0] = w - (bbox[0] + bbox[2])
                                elif self.direction == 'vertical':
                                    bbox[1] = h - (bbox[1] + bbox[3])
                            
                            instance['bbox'] = bbox.tolist()
        
        return results