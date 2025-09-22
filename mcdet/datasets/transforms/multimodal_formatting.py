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
        """
        [수정된 최종 버전]
        Ground truth 인스턴스를 포맷하며, 인스턴스가 없는 경우에도
        항상 올바른 형태의 빈 텐서를 생성하여 안정성을 보장합니다.
        """
        gt_instances = InstanceData()
        
        # results에 'instances' 키가 있고, 그 리스트가 비어있지 않은 경우
        if 'instances' in results and results['instances']:
            bboxes = []
            labels = []
            for instance in results['instances']:
                if 'bbox' in instance:
                    bboxes.append(instance['bbox'])
                if 'bbox_label' in instance:
                    labels.append(instance['bbox_label'])

            # bboxes 리스트가 비어있지 않으면 텐서로 변환, 비어있으면 빈 텐서 생성
            if bboxes:
                gt_instances.bboxes = torch.tensor(bboxes, dtype=torch.float32)
            else:
                gt_instances.bboxes = torch.empty((0, 4), dtype=torch.float32)

            # labels 리스트가 비어있지 않으면 텐서로 변환, 비어있으면 빈 텐서 생성
            if labels:
                gt_instances.labels = torch.tensor(labels, dtype=torch.long)
            else:
                gt_instances.labels = torch.empty((0,), dtype=torch.long)

        # 'instances' 키가 없거나, 리스트가 처음부터 비어있는 경우
        else:
            # 모델의 loss 함수가 에러를 일으키지 않도록
            # .bboxes와 .labels 속성을 명시적으로 생성해줍니다.
            gt_instances.bboxes = torch.empty((0, 4), dtype=torch.float32)
            gt_instances.labels = torch.empty((0,), dtype=torch.long)
            
            # 만약 mask 등 다른 GT 데이터를 사용한다면, 해당 키에 대해서도
            # 빈 객체를 생성해주는 코드를 추가해야 할 수 있습니다.

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

