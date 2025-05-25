# mcdet/models/detectors/deliver_detector.py

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional, Tuple
import numpy as np

from mmdet.registry import MODELS
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.utils import multi_apply, unpack_gt_instances
from mmdet.structures import DetDataSample, SampleList, OptSampleList
from mmengine.structures import InstanceData
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class DELIVERDetector(BaseDetector):
    """Complete DELIVER multimodal object detector.
    
    This detector processes multimodal inputs (RGB, Depth, Event, LiDAR) 
    and performs object detection using FasterRCNN-style architecture.
    
    Architecture:
    Input: List[rgb, depth, event, lidar] -> CMNext Backbone -> FPN -> RPN -> ROI Head -> Output
    """
    
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )
        
        # Build backbone (CMNext)
        self.backbone = MODELS.build(backbone)
        
        # Build neck (FPN)
        if neck is not None:
            self.neck = MODELS.build(neck)
        
        # Build RPN head
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = MODELS.build(rpn_head_)
        
        # Build ROI head
        if roi_head is not None:
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None
    
    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None
    
    def extract_feat(self, batch_inputs: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Extract features from multimodal inputs.
        
        Args:
            batch_inputs: List of multimodal tensors [rgb, depth, event, lidar]
                         Each tensor has shape (B, C, H, W)
        
        Returns:
            Tuple of multi-scale features from FPN
        """
        # Extract backbone features
        x = self.backbone(batch_inputs)
        
        # Apply neck (FPN) if present
        if self.with_neck:
            x = self.neck(x)
        
        return x
    
    def loss(self, 
             batch_inputs: List[torch.Tensor], 
             batch_data_samples: SampleList) -> Dict[str, torch.Tensor]:
        """Calculate losses from a batch of inputs and data samples.
        
        Args:
            batch_inputs: List of multimodal input tensors
            batch_data_samples: List of data samples containing annotations
            
        Returns:
            Dict of losses
        """
        # Extract features
        x = self.extract_feat(batch_inputs)
        
        losses = dict()
        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            
            # RPN losses
            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg
            )
            
            # Update proposal into data_samples
            for data_sample, rpn_results in zip(rpn_data_samples, rpn_results_list):
                data_sample.proposals = rpn_results
            
            losses.update(rpn_losses)
        else:
            # Use ground truth bboxes as proposals
            rpn_data_samples = batch_data_samples
            for data_sample in rpn_data_samples:
                data_sample.proposals = data_sample.gt_instances
        
        # ROI head forward and loss
        if self.with_roi_head:
            roi_losses = self.roi_head.loss(x, rpn_data_samples)
            losses.update(roi_losses)
        
        return losses
    
    def predict(self,
                batch_inputs: List[torch.Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples.
        
        Args:
            batch_inputs: List of multimodal input tensors
            batch_data_samples: List of data samples
            rescale: Whether to rescale the results to original image size
            
        Returns:
            List of detection results
        """
        # Extract features
        x = self.extract_feat(batch_inputs)
        
        # RPN forward
        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False
            )
        else:
            # Use ground truth bboxes as proposals for testing
            rpn_results_list = [
                data_sample.gt_instances for data_sample in batch_data_samples
            ]
        
        # Update proposals into data_samples
        for data_sample, rpn_results in zip(batch_data_samples, rpn_results_list):
            data_sample.proposals = rpn_results
        
        # ROI head forward
        if self.with_roi_head:
            results_list = self.roi_head.predict(
                x, batch_data_samples, rescale=rescale
            )
        else:
            results_list = rpn_results_list
        
        # Update predictions into data_samples
        for data_sample, pred_instances in zip(batch_data_samples, results_list):
            data_sample.pred_instances = pred_instances
        
        return batch_data_samples
    
    def _forward(self,
                 batch_inputs: List[torch.Tensor],
                 batch_data_samples: OptSampleList = None) -> Tuple[List[torch.Tensor]]:
        """Network forward process.
        
        Args:
            batch_inputs: List of multimodal input tensors
            batch_data_samples: List of data samples
            
        Returns:
            Tuple of network outputs
        """
        x = self.extract_feat(batch_inputs)
        results = self.rpn_head.forward(x) if self.with_rpn else ()
        return results


@MODELS.register_module()
class DELIVERFasterRCNN(DELIVERDetector):
    """DELIVER FasterRCNN - Multimodal Faster R-CNN implementation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@MODELS.register_module()
class DELIVERRetinaNet(DELIVERDetector):
    """DELIVER RetinaNet - Multimodal RetinaNet implementation.
    
    Single-stage detector version for DELIVER dataset.
    """
    
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        
        # Initialize base detector without RPN
        super(DELIVERDetector, self).__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )
        
        self.backbone = MODELS.build(backbone)
        
        if neck is not None:
            self.neck = MODELS.build(neck)
        
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    @property
    def with_bbox(self) -> bool:
        """bool: whether the detector has a bbox head"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None
    
    def loss(self, 
             batch_inputs: List[torch.Tensor], 
             batch_data_samples: SampleList) -> Dict[str, torch.Tensor]:
        """Calculate losses for RetinaNet."""
        x = self.extract_feat(batch_inputs)
        
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses
    
    def predict(self,
                batch_inputs: List[torch.Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results for RetinaNet."""
        x = self.extract_feat(batch_inputs)
        
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale
        )
        
        for data_sample, pred_instances in zip(batch_data_samples, results_list):
            data_sample.pred_instances = pred_instances
        
        return batch_data_samples


# Utility functions for data handling
def stack_multimodal_batch(batch_list: List[Dict]) -> Dict:
    """Stack a list of multimodal samples into batch format.
    
    Args:
        batch_list: List of samples from dataloader
        
    Returns:
        Batched dictionary with multimodal inputs
    """
    if not batch_list:
        return {}
    
    # Extract multimodal inputs
    multimodal_inputs = []
    data_samples = []
    
    for sample in batch_list:
        multimodal_inputs.append(sample['inputs'])  # List of modality tensors
        data_samples.append(sample['data_samples'])
    
    # Stack each modality separately
    num_modalities = len(multimodal_inputs[0])
    batched_inputs = []
    
    for modality_idx in range(num_modalities):
        modality_batch = [inputs[modality_idx] for inputs in multimodal_inputs]
        stacked_modality = torch.stack(modality_batch, dim=0)
        batched_inputs.append(stacked_modality)
    
    return {
        'inputs': batched_inputs,
        'data_samples': data_samples
    }


# Data preprocessor for DELIVER
@MODELS.register_module()
class DELIVERDataPreprocessor(nn.Module):
    """Data preprocessor for DELIVER multimodal detection.
    
    Handles normalization and batching of multimodal inputs.
    """
    
    def __init__(self,
                 mean: List[List[float]] = None,
                 std: List[List[float]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: float = 0.0):
        super().__init__()
        
        # Default normalization values for each modality (following DELIVER original)
        if mean is None:
            self.mean = [
                [0.485, 0.456, 0.406],       # RGB (ImageNet standard - same as DELIVER)
                [0.0, 0.0, 0.0],             # Depth
                [0.0, 0.0, 0.0],             # Event
                [0.0, 0.0, 0.0]              # LiDAR
            ]
        else:
            self.mean = mean
            
        if std is None:
            self.std = [
                [0.229, 0.224, 0.225],       # RGB (ImageNet standard - same as DELIVER)
                [1.0, 1.0, 1.0],             # Depth
                [1.0, 1.0, 1.0],             # Event
                [1.0, 1.0, 1.0]              # LiDAR
            ]
        else:
            self.std = std
        
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        
        # Register normalization parameters
        for i, (m, s) in enumerate(zip(self.mean, self.std)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, -1, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, -1, 1, 1))
    
    def forward(self, data: Dict, training: bool = False) -> Dict:
        """Preprocess multimodal data.
        
        Args:
            data: Input data dictionary
            training: Whether in training mode
            
        Returns:
            Preprocessed data dictionary
        """
        inputs = data['inputs']  # List of multimodal tensors
        
        # Normalize each modality
        normalized_inputs = []
        for i, modal_tensor in enumerate(inputs):
            # Convert to float and normalize
            modal_tensor = modal_tensor.float()
            mean = getattr(self, f'mean_{i}')
            std = getattr(self, f'std_{i}')
            
            normalized_modal = (modal_tensor - mean) / std
            normalized_inputs.append(normalized_modal)
        
        # Apply padding if needed
        if self.pad_size_divisor > 1:
            normalized_inputs = self._pad_inputs(normalized_inputs)
        
        data['inputs'] = normalized_inputs
        return data
    
    def _pad_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Pad inputs to be divisible by pad_size_divisor."""
        padded_inputs = []
        
        for modal_tensor in inputs:
            B, C, H, W = modal_tensor.shape
            
            # Calculate padding
            pad_h = (self.pad_size_divisor - H % self.pad_size_divisor) % self.pad_size_divisor
            pad_w = (self.pad_size_divisor - W % self.pad_size_divisor) % self.pad_size_divisor
            
            if pad_h > 0 or pad_w > 0:
                modal_tensor = torch.nn.functional.pad(
                    modal_tensor, (0, pad_w, 0, pad_h), value=self.pad_value
                )
            
            padded_inputs.append(modal_tensor)
        
        return padded_inputs