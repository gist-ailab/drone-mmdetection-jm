# mcdet/datasets/transforms/multimodal_formatting.py

import torch
import numpy as np
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from typing import Dict, List, Union, Optional, Sequence
from mmdet.registry import DATA_SAMPLERS

@TRANSFORMS.register_module()
class PackDELIVERDetInputs:
    """Pack DELIVER multimodal detection inputs with proper tensor formatting."""
    
    def __init__(self,
                 meta_keys: Sequence[str] = (
                     'img_id', 'img_path', 'ori_shape', 'img_shape',
                     'scale_factor', 'flip', 'flip_direction',
                     'modality_paths'
                 )):
        self.meta_keys = meta_keys
    
    def _format_gt_instances(self, results: Dict) -> InstanceData:
        """Format ground truth instances with proper tensor types."""
        gt_instances = InstanceData()
        if 'instances' in results:
            bboxes = []
            labels = []
            for instance in results['instances']:
                if 'bbox' in instance:
                    bboxes.append(instance['bbox'])
                if 'bbox_label' in instance:
                    labels.append(instance['bbox_label'])
            if bboxes:                          # Ensure bboxes are float32 tensors
                gt_instances.bboxes = torch.tensor(bboxes, dtype=torch.float32)            
            if labels:                  # Ensure labels are long tensors    
                gt_instances.labels = torch.tensor(labels, dtype=torch.long)
        
        return gt_instances
    
    def __call__(self, results: Dict) -> Dict:
        """Pack multimodal detection inputs with device-aware GT handling."""
        packed_results = {}
        # Format multimodal images
        if isinstance(results['img'], list):
            inputs = self._format_multimodal_images(results['img'])
        else:
            # Single modal fallback
            img = results['img']
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)
            if img.dtype != torch.float32:
                img = img.float()
            if len(img.shape) == 3:
                img = img.permute(2, 0, 1)
            inputs = [img]
        
        packed_results['inputs'] = inputs
        
        # Create DetDataSample
        data_sample = DetDataSample()
        
        # Format ground truth - 여기서 올바른 텐서 생성
        if any(key in results for key in ['instances', 'gt_bboxes', 'gt_labels']):
            gt_instances = self._format_gt_instances(results)
            data_sample.gt_instances = gt_instances
        
        # Format meta information
        metainfo = self._format_metainfo(results)
        h, w = inputs[0].shape[-2:]  # 수정된 부분
        if 'pad_shape' not in metainfo:
            metainfo['pad_shape'] = (h, w)
        data_sample.set_metainfo(metainfo)
        
        packed_results['data_samples'] = data_sample
        return packed_results
    
    # 기타 메서드들은 기존과 동일...
    def _format_multimodal_images(self, imgs: List[np.ndarray]) -> List[torch.Tensor]:
        """Convert list of numpy images to list of torch tensors."""
        formatted_imgs = []
        
        for img in imgs:
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)
            
            if not isinstance(img, torch.Tensor):
                if img.strides and any(s < 0 for s in img.strides):
                    img = img.copy()
                img = torch.from_numpy(img)
            
            if img.dtype != torch.float32:
                img = img.float()
            
            if len(img.shape) == 3:
                img = img.permute(2, 0, 1)
            
            formatted_imgs.append(img)
        
        return formatted_imgs
    
    def _format_metainfo(self, results: Dict) -> Dict:
        """Format meta information."""
        metainfo = {}
        
        for key in self.meta_keys:
            if key in results:
                metainfo[key] = results[key]
        
        if 'modality_paths' in results:
            metainfo['modality_paths'] = results['modality_paths']
        
        if 'img_path' in results and isinstance(results['img_path'], list):
            metainfo['img_path'] = results['img_path']
        elif 'modality_paths' in results:
            metainfo['img_path'] = [
                results['modality_paths']['rgb'],
                results['modality_paths']['depth'],
                results['modality_paths']['event'],
                results['modality_paths']['lidar']
            ]
        
        return metainfo

