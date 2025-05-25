# mcdet/datasets/transforms/deliver_transforms.py

import cv2
import mmcv
import numpy as np
import random
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import Resize, RandomCrop, RandomFlip, Normalize
from typing import Dict, List, Tuple, Union, Optional
import copy


@TRANSFORMS.register_module()
class DELIVERResize:
    """Resize transform for DELIVER multimodal images.
    
    Applies same resize parameters to all modalities (RGB, Depth, Event, LiDAR)
    and updates bboxes accordingly.
    """
    
    def __init__(self, 
                 img_scale: Union[Tuple[int, int], List[Tuple[int, int]]] = None,
                 multiscale_mode: str = 'range',
                 ratio_range: Tuple[float, float] = None,
                 keep_ratio: bool = True,
                 bbox_clip_border: bool = True,
                 backend: str = 'cv2',
                 interpolation: str = 'bilinear'):
        
        self.img_scale = img_scale
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.bbox_clip_border = bbox_clip_border
        self.backend = backend
        self.interpolation = interpolation
    
    def _get_target_scale(self, ori_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Get target scale for resizing."""
        if isinstance(self.img_scale, list):
            if self.multiscale_mode == 'range':
                target_scale = random.choice(self.img_scale)
            else:  # 'value'
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
    
    def _resize_bboxes(self, bboxes: np.ndarray, scale_factor: Tuple[float, float]) -> np.ndarray:
        """Resize bboxes according to scale factor."""
        if len(bboxes) == 0:
            return bboxes
            
        bboxes = bboxes.copy()
        bboxes[:, 0::2] *= scale_factor[0]  # x coordinates
        bboxes[:, 1::2] *= scale_factor[1]  # y coordinates
        
        if self.bbox_clip_border:
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, new_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, new_shape[0])
            
        return bboxes
    
    def __call__(self, results: Dict) -> Dict:
        """Apply resize to all modalities."""
        if isinstance(results['img'], list):
            # Get original shape from first modality (RGB)
            ori_shape = results['img'][0].shape[:2]  # (H, W)
            
            # Determine target scale (same for all modalities)
            target_scale = self._get_target_scale(ori_shape)
            
            if self.keep_ratio:
                new_shape, scale_factor = mmcv.rescale_size(ori_shape, target_scale, return_scale=True)
            else:
                new_shape = target_scale
                scale_factor = (target_scale[1] / ori_shape[1], target_scale[0] / ori_shape[0])
            
            # Resize all modality images
            resized_imgs = []
            for img in results['img']:
                resized_img = self._resize_img(img, new_shape)
                resized_imgs.append(resized_img)
            
            results['img'] = resized_imgs
            results['img_shape'] = [new_shape + (img.shape[2],) for img in resized_imgs]
            results['scale_factor'] = scale_factor
            
            # Resize bboxes if present
            if 'gt_bboxes' in results:
                results['gt_bboxes'] = self._resize_bboxes(results['gt_bboxes'], scale_factor)
            
            # Update instance bboxes
            if 'instances' in results:
                for instance in results['instances']:
                    if 'bbox' in instance:
                        bbox = np.array(instance['bbox']).reshape(1, -1)
                        resized_bbox = self._resize_bboxes(bbox, scale_factor)
                        instance['bbox'] = resized_bbox.flatten().tolist()
        
        return results


@TRANSFORMS.register_module()
class DELIVERRandomCrop:
    """Random crop transform for DELIVER multimodal images."""
    
    def __init__(self, 
                 crop_size: Tuple[int, int],
                 crop_type: str = 'absolute',
                 allow_negative_crop: bool = False,
                 bbox_clip_border: bool = True):
        
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
    
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
    
    def _crop_bboxes(self, bboxes: np.ndarray, crop_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop bboxes according to crop region."""
        if len(bboxes) == 0:
            return bboxes
            
        x1, y1, x2, y2 = crop_bbox
        
        # Shift bboxes
        bboxes = bboxes.copy()
        bboxes[:, 0::2] -= x1  # x coordinates
        bboxes[:, 1::2] -= y1  # y coordinates
        
        if self.bbox_clip_border:
            crop_w, crop_h = x2 - x1, y2 - y1
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, crop_w)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, crop_h)
            
        return bboxes
    
    def __call__(self, results: Dict) -> Dict:
        """Apply crop to all modalities."""
        if isinstance(results['img'], list):
            # Get crop bbox (same for all modalities)
            img_shape = results['img'][0].shape
            crop_bbox = self._get_crop_bbox(img_shape)
            
            # Crop all modality images
            cropped_imgs = []
            for img in results['img']:
                cropped_img = self._crop_img(img, crop_bbox)
                cropped_imgs.append(cropped_img)
            
            results['img'] = cropped_imgs
            results['img_shape'] = [img.shape for img in cropped_imgs]
            
            # Crop bboxes if present
            if 'gt_bboxes' in results:
                results['gt_bboxes'] = self._crop_bboxes(results['gt_bboxes'], crop_bbox)
            
            # Update instance bboxes
            if 'instances' in results:
                for instance in results['instances']:
                    if 'bbox' in instance:
                        bbox = np.array(instance['bbox']).reshape(1, -1)
                        cropped_bbox = self._crop_bboxes(bbox, crop_bbox)
                        instance['bbox'] = cropped_bbox.flatten().tolist()
        
        return results


@TRANSFORMS.register_module()
class DELIVERRandomFlip:
    """Random flip transform for DELIVER multimodal images."""
    
    def __init__(self, 
                 prob: float = 0.5,
                 direction: str = 'horizontal'):
        
        self.prob = prob
        self.direction = direction
    
    def _flip_img(self, img: np.ndarray, direction: str) -> np.ndarray:
        """Flip single image."""
        return mmcv.imflip(img, direction=direction)
    
    def _flip_bboxes(self, bboxes: np.ndarray, img_shape: Tuple[int, int], direction: str) -> np.ndarray:
        """Flip bboxes according to flip direction."""
        if len(bboxes) == 0:
            return bboxes
            
        h, w = img_shape[:2]
        bboxes = bboxes.copy()
        
        if direction == 'horizontal':
            bboxes[:, 0::4] = w - bboxes[:, 2::4]  # x1 = w - x2
            bboxes[:, 2::4] = w - bboxes[:, 0::4]  # x2 = w - x1
        elif direction == 'vertical':
            bboxes[:, 1::4] = h - bboxes[:, 3::4]  # y1 = h - y2
            bboxes[:, 3::4] = h - bboxes[:, 1::4]  # y2 = h - y1
            
        return bboxes
    
    def __call__(self, results: Dict) -> Dict:
        """Apply flip to all modalities."""
        if isinstance(results['img'], list):
            # Decide whether to flip (same for all modalities)
            flip = random.random() < self.prob
            
            if flip:
                img_shape = results['img'][0].shape
                
                # Flip all modality images
                flipped_imgs = []
                for img in results['img']:
                    flipped_img = self._flip_img(img, self.direction)
                    flipped_imgs.append(flipped_img)
                
                results['img'] = flipped_imgs
                
                # Flip bboxes if present
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = self._flip_bboxes(results['gt_bboxes'], img_shape, self.direction)
                
                # Update instance bboxes
                if 'instances' in results:
                    for instance in results['instances']:
                        if 'bbox' in instance:
                            bbox = np.array(instance['bbox']).reshape(1, -1)
                            flipped_bbox = self._flip_bboxes(bbox, img_shape, self.direction)
                            instance['bbox'] = flipped_bbox.flatten().tolist()
        
        return results


@TRANSFORMS.register_module()
class DELIVERNormalize:
    def __init__(self, mean=None, std=None, to_rgb=True):
        if mean is None:
            # DELIVER 원본과 동일한 값 사용
            self.mean = [
                [0.485, 0.456, 0.406],   # RGB (ImageNet - DELIVER standard)
                [0, 0, 0],               # Depth
                [0, 0, 0],               # Event
                [0, 0, 0]                # LiDAR
            ]
        else:
            self.mean = mean
            
        if std is None:
            self.std = [
                [0.229, 0.224, 0.225],   # RGB (ImageNet - DELIVER standard)
                [1, 1, 1],               # Depth
                [1, 1, 1],               # Event  
                [1, 1, 1]                # LiDAR
            ]
        else:
            self.std = std
            
        self.to_rgb = to_rgb
    
    def _normalize_img(self, img: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
        """Normalize single image."""
        return mmcv.imnormalize(img, np.array(mean), np.array(std), self.to_rgb)
    
    def __call__(self, results: Dict) -> Dict:
        """Apply normalization to all modalities."""
        if isinstance(results['img'], list):
            normalized_imgs = []
            
            for i, img in enumerate(results['img']):
                mean = self.mean[i] if i < len(self.mean) else self.mean[-1]
                std = self.std[i] if i < len(self.std) else self.std[-1]
                
                normalized_img = self._normalize_img(img, mean, std)
                normalized_imgs.append(normalized_img)
            
            results['img'] = normalized_imgs
            results['img_norm_cfg'] = {
                'mean': self.mean,
                'std': self.std,
                'to_rgb': self.to_rgb
            }
        
        return results