# mcdet/datasets/transforms/multimodal_formatting.py

import torch
import numpy as np
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from typing import Dict, List, Union, Optional, Sequence


@TRANSFORMS.register_module()
class PackDELIVERDetInputs:
    """Pack DELIVER multimodal detection inputs.
    
    This transform converts preprocessed multimodal images and annotations
    to the format expected by DELIVER detection models.
    
    Final format:
    - inputs: List[torch.Tensor] with shape [B, C, H, W] for each modality
    - data_samples: List[DetDataSample] containing ground truth and metadata
    """
    
    def __init__(self,
                 meta_keys: Sequence[str] = (
                     'img_id', 'img_path', 'ori_shape', 'img_shape',
                     'scale_factor', 'flip', 'flip_direction',
                     'modality_paths'
                 )):
        """
        Args:
            meta_keys: Meta information keys to be saved in DetDataSample
        """
        self.meta_keys = meta_keys
    
    def _format_multimodal_images(self, imgs: List[np.ndarray]) -> List[torch.Tensor]:
        """Convert list of numpy images to list of torch tensors.
        
        Args:
            imgs: List of images with shape (H, W, C)
            
        Returns:
            List of tensors with shape (C, H, W)
        """
        formatted_imgs = []
        
        for img in imgs:
            # Convert numpy to tensor
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)
            
            # Ensure float32
            if img.dtype != torch.float32:
                img = img.float()
            
            # Convert HWC to CHW
            if len(img.shape) == 3:
                img = img.permute(2, 0, 1)
            
            formatted_imgs.append(img)
        
        return formatted_imgs
    
    def _format_gt_instances(self, results: Dict) -> InstanceData:
        """Format ground truth instances.
        
        Args:
            results: Results dict containing ground truth data
            
        Returns:
            InstanceData containing formatted ground truth
        """
        gt_instances = InstanceData()
        
        # Process bboxes
        if 'instances' in results:
            bboxes = []
            labels = []
            
            for instance in results['instances']:
                if 'bbox' in instance:
                    bboxes.append(instance['bbox'])
                if 'bbox_label' in instance:
                    labels.append(instance['bbox_label'])
            
            if bboxes:
                gt_instances.bboxes = torch.tensor(bboxes, dtype=torch.float32)
            if labels:
                gt_instances.labels = torch.tensor(labels, dtype=torch.long)
        
        # Legacy format support
        elif 'gt_bboxes' in results:
            gt_instances.bboxes = torch.tensor(results['gt_bboxes'], dtype=torch.float32)
        
        if 'gt_labels' in results:
            gt_instances.labels = torch.tensor(results['gt_labels'], dtype=torch.long)
        
        # Process masks if present
        if 'gt_masks' in results:
            # Handle masks if needed for segmentation
            pass
        
        return gt_instances
    
    def _format_metainfo(self, results: Dict) -> Dict:
        """Format meta information.
        
        Args:
            results: Results dict
            
        Returns:
            Dict containing meta information
        """
        metainfo = {}
        
        for key in self.meta_keys:
            if key in results:
                metainfo[key] = results[key]
        
        # Add special handling for multimodal paths
        if 'modality_paths' in results:
            metainfo['modality_paths'] = results['modality_paths']
        
        # Ensure img_path is list for multimodal
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
    
    def __call__(self, results: Dict) -> Dict:
        """Pack multimodal detection inputs.
        
        Args:
            results: Results dict from previous transforms
            
        Returns:
            Dict with packed inputs and data samples
        """
        packed_results = {}
        
        # 1. Format multimodal images
        if isinstance(results['img'], list):
            # Multimodal case
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
            inputs = [img]  # Wrap in list for consistency
        
        packed_results['inputs'] = inputs
        
        # 2. Create DetDataSample
        data_sample = DetDataSample()
        
        # Format ground truth
        if any(key in results for key in ['instances', 'gt_bboxes', 'gt_labels']):
            gt_instances = self._format_gt_instances(results)
            data_sample.gt_instances = gt_instances
        
        # Format meta information
        metainfo = self._format_metainfo(results)
        data_sample.set_metainfo(metainfo)
        
        packed_results['data_samples'] = data_sample
        
        return packed_results


@TRANSFORMS.register_module()
class PackDELIVERDetInputsBatch:
    """Batch version of PackDELIVERDetInputs for DataLoader collate_fn.
    
    This handles batching of multimodal detection data.
    """
    
    def __init__(self):
        pass
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch of DELIVER detection data.
        
        Args:
            batch: List of samples from PackDELIVERDetInputs
            
        Returns:
            Batched dict with:
            - inputs: List[torch.Tensor] with shape [B, C, H, W] for each modality
            - data_samples: List[DetDataSample]
        """
        if not batch:
            return {}
        
        # Extract inputs and data_samples
        batch_inputs = [sample['inputs'] for sample in batch]
        batch_data_samples = [sample['data_samples'] for sample in batch]
        
        # Stack multimodal inputs
        num_modalities = len(batch_inputs[0])
        stacked_inputs = []
        
        for modality_idx in range(num_modalities):
            modality_batch = [inputs[modality_idx] for inputs in batch_inputs]
            stacked_modality = torch.stack(modality_batch, dim=0)
            stacked_inputs.append(stacked_modality)
        
        return {
            'inputs': stacked_inputs,
            'data_samples': batch_data_samples
        }


# Additional utility functions
def deliver_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for DELIVER dataset.
    
    This function is used by DataLoader to batch samples.
    """
    batch_packer = PackDELIVERDetInputsBatch()
    return batch_packer(batch)


# For compatibility with MMDetection's pseudo_collate
@TRANSFORMS.register_module()
class PseudoDELIVERCollate:
    """Pseudo collate for single sample (used in testing)."""
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Handle single sample case."""
        if len(batch) == 1:
            sample = batch[0]
            # Add batch dimension
            if 'inputs' in sample:
                inputs = sample['inputs']
                batched_inputs = [tensor.unsqueeze(0) for tensor in inputs]
                sample['inputs'] = batched_inputs
            
            if 'data_samples' in sample:
                sample['data_samples'] = [sample['data_samples']]
            
            return sample
        else:
            return deliver_collate_fn(batch)