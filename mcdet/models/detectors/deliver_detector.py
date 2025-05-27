# mcdet/models/detectors/deliver_detector.py

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional, Tuple
import numpy as np
import logging

from mmdet.registry import MODELS
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.utils import multi_apply, unpack_gt_instances
from mmdet.structures import DetDataSample, SampleList, OptSampleList
from mmengine.structures import InstanceData
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
import cv2
# Setup logger for optional debugging
logger = logging.getLogger(__name__)


def convert(t):
    t = t.cpu().numpy()
    t= t.transpose(1,2,0)
    cv2.imwrite('tmp.png', t)

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
    
    def _ensure_metainfo(self, batch_inputs: List[torch.Tensor], batch_data_samples: SampleList) -> SampleList:
        """Ensure all data samples have required metainfo for MMDetection.
        
        Args:
            batch_inputs: List of input tensors
            batch_data_samples: List of data samples
            
        Returns:
            Updated data samples with guaranteed metainfo
        """
        # Get image shape from first input tensor
        if batch_inputs and len(batch_inputs) > 0:
            img_shape = batch_inputs[0].shape[2:]  # (H, W)
        else:
            img_shape = (512, 512)  # Default fallback
        
        for i, data_sample in enumerate(batch_data_samples):
            # Ensure metainfo exists
            if not hasattr(data_sample, 'metainfo') or data_sample.metainfo is None:
                data_sample.set_metainfo({})
            
            metainfo = data_sample.metainfo
            
            # Set default values if missing
            if 'ori_shape' not in metainfo:
                metainfo['ori_shape'] = img_shape
            
            if 'img_shape' not in metainfo:
                metainfo['img_shape'] = img_shape
            
            if 'pad_shape' not in metainfo:
                metainfo['pad_shape'] = img_shape
            
            if 'scale_factor' not in metainfo:
                metainfo['scale_factor'] = (1.0, 1.0)
            
            # Additional useful metainfo
            metainfo['pad_size_divisor'] = 32
            metainfo['batch_input_shape'] = img_shape
        
        return batch_data_samples
    
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
        # Ensure metainfo before any processing
        batch_data_samples = self._ensure_metainfo(batch_inputs, batch_data_samples)
        
        # Extract features
        x = self.extract_feat(batch_inputs)
        losses = dict()
        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            
            # Emergency fallback for metainfo
            for sample in rpn_data_samples:
                if not hasattr(sample, 'metainfo') or sample.metainfo is None:
                    sample.set_metainfo({
                        'ori_shape': batch_inputs[0].shape[2:],
                        'img_shape': batch_inputs[0].shape[2:], 
                        'pad_shape': batch_inputs[0].shape[2:],
                        'scale_factor': (1.0, 1.0)
                    })
            
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
            rpn_results_list = []
            for data_sample in rpn_data_samples:
                data_sample.proposals = data_sample.gt_instances
                rpn_results_list.append(data_sample.gt_instances)
        
        # ROI head forward and loss
        if self.with_roi_head:
            # Check if roi_head.loss expects 2 or 3 arguments
            import inspect
            roi_loss_signature = inspect.signature(self.roi_head.loss)
            num_params = len([p for p in roi_loss_signature.parameters.values() 
                            if p.kind != p.VAR_KEYWORD and p.name != 'self'])
            
            if num_params == 2:
                roi_losses = self.roi_head.loss(x, rpn_data_samples)
            elif num_params >= 3:
                roi_losses = self.roi_head.loss(x, rpn_results_list, rpn_data_samples)
            else:
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
        # Ensure metainfo for prediction as well
        batch_data_samples = self._ensure_metainfo(batch_inputs, batch_data_samples)
        
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
                x, rpn_results, batch_data_samples, rescale=rescale
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
        # Ensure metainfo
        batch_data_samples = self._ensure_metainfo(batch_inputs, batch_data_samples)
        
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses
    
    def predict(self,
                batch_inputs: List[torch.Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results for RetinaNet."""
        # Ensure metainfo
        batch_data_samples = self._ensure_metainfo(batch_inputs, batch_data_samples)
        
        x = self.extract_feat(batch_inputs)
        
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale
        )
        
        for data_sample, pred_instances in zip(batch_data_samples, results_list):
            data_sample.pred_instances = pred_instances
        
        return batch_data_samples


# # Utility functions for data handling
# def stack_multimodal_batch(batch_list: List[Dict]) -> Dict:
#     """Stack a list of multimodal samples into batch format.
    
#     Args:
#         batch_list: List of samples from dataloader
        
#     Returns:
#         Batched dictionary with multimodal inputs
#     """
#     if not batch_list:
#         return {}
    
#     # Extract multimodal inputs
#     multimodal_inputs = []
#     data_samples = []
    
#     for sample in batch_list:
#         multimodal_inputs.append(sample['inputs'])  # List of modality tensors
#         data_samples.append(sample['data_samples'])
    
#     # Stack each modality separately
#     num_modalities = len(multimodal_inputs[0])
#     batched_inputs = []
    
#     for modality_idx in range(num_modalities):
#         modality_batch = [inputs[modality_idx] for inputs in multimodal_inputs]
#         stacked_modality = torch.stack(modality_batch, dim=0)
#         batched_inputs.append(stacked_modality)
    
#     return {
#         'inputs': batched_inputs,
#         'data_samples': data_samples
#     }


@MODELS.register_module()
class DELIVERDataPreprocessor(nn.Module):
    """Data preprocessor with proper shape handling for DELIVER multimodal detection."""
    
    def __init__(self,
                 mean: List[List[float]] = None,
                 std: List[List[float]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: float = 0.0,
                 to_rgb: bool = True,
                 bgr_to_rgb: bool = False):
        super().__init__()
        
        # Default normalization values for each modality
        if mean is None:
            self.mean = [
                [0.485, 0.456, 0.406],       # RGB
                [0.0, 0.0, 0.0],             # Depth
                [0.0, 0.0, 0.0],             # Event
                [0.0, 0.0, 0.0]              # LiDAR
            ]
        else:
            self.mean = mean
            
        if std is None:
            self.std = [
                [0.229, 0.224, 0.225],       # RGB
                [1.0, 1.0, 1.0],             # Depth
                [1.0, 1.0, 1.0],             # Event
                [1.0, 1.0, 1.0]              # LiDAR
            ]
        else:
            self.std = std
        
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb
        self.bgr_to_rgb = bgr_to_rgb
        
        # Register normalization parameters as buffers
        for i, (m, s) in enumerate(zip(self.mean, self.std)):
            self.register_buffer(
                f'mean_{i}', 
                torch.tensor(m, dtype=torch.float32).view(1, -1, 1, 1)
            )
            self.register_buffer(
                f'std_{i}', 
                torch.tensor(s, dtype=torch.float32).view(1, -1, 1, 1)
            )
    def _unwrap_data(self, data):
        """Recursively unwrap tuples from data."""
        if isinstance(data, tuple) and len(data) == 1:
            return self._unwrap_data(data[0])
        elif isinstance(data, list):
            return [self._unwrap_data(item) for item in data]
        else:
            return data
    

    def forward(self, data: Dict, training: bool = False) -> Dict:
        """Preprocess with proper shape validation."""
        inputs = data['inputs']
        data_samples = data.get('data_samples', [])
        # Get target device
        target_device = next(self.buffers()).device
        inputs = self._unwrap_data(inputs)  # Unwrap any nested tuples
        input_list = []
        for idx in range(len(inputs)):
            modal_data = inputs[idx][0]
            batched_data = modal_data.unsqueeze(0)
            input_list.append(batched_data)

        inputs = input_list
        
        # 2. Normalize and move to device
        original_shape = inputs[0].shape[2:]  # torch.Size([H, W])
        original_shape = tuple(int(x) for x in original_shape)  # Convert to (H, W) tuple


        # Normalization
        normalized_inputs = []
        for i, modal_tensor in enumerate(inputs):
            # Move to target device and convert to float
            modal_tensor = modal_tensor.float().to(target_device)

            # 🔥 핵심 수정: 모달리티별 다른 정규화 방식 적용
            if i == 0:  # RGB 모달리티
                # 1단계: [0, 255] -> [0, 1] 변환
                modal_tensor = modal_tensor / 255.0
                
                # 2단계: ImageNet 정규화
                mean = getattr(self, f'mean_{i}')
                std = getattr(self, f'std_{i}')
                normalized_modal = (modal_tensor - mean) / std
                
            else:  # Depth, Event, LiDAR 모달리티
                # 각 모달리티별로 적절한 정규화 적용
                if i == 1:  # Depth
                    modal_tensor = modal_tensor / 255.0    
                    # Event 데이터 정규화 (보통 이미 [-1, 1] 범위)
                    modal_tensor = modal_tensor / 255.0  
                    # LiDAR 데이터 정규화 (예: 0-255 -> 0-1)
                    modal_tensor = modal_tensor / 255.0 
                # Mean/Std 정규화 (간단한 z-score 또는 그대로 사용)
                mean = getattr(self, f'mean_{i}')
                std = getattr(self, f'std_{i}')
                normalized_modal = (modal_tensor - mean) / std
            normalized_inputs.append(normalized_modal)

        # 3. Apply padding if needed
        if self.pad_size_divisor > 1:
            normalized_inputs = self._pad_inputs(normalized_inputs)
        # Get final shape
        final_shape = normalized_inputs[0].shape[2:]  # torch.Size([H, W])
        final_shape = tuple(int(x) for x in final_shape)  # Convert to (H, W) tuple
        # 4. Update metadata and move GT data
        for data_sample in data_samples:
            self._fix_and_validate_metainfo(data_sample, original_shape, final_shape)
            self._move_gt_to_device(data_sample, target_device)

        cat_list=[]
        cat_list.append(normalized_inputs)
        print(len(cat_list))

        data['inputs'] = normalized_inputs
        data['data_samples'] = data_samples
        return data
    
    def _fix_and_validate_metainfo(self, data_sample, original_shape, final_shape):
        """메타정보 검증 및 수정 - Shape 타입 문제 해결"""
        
        # metainfo 존재 확인
        if not hasattr(data_sample, 'metainfo') or data_sample.metainfo is None:
            data_sample.set_metainfo({})
        
        metainfo = data_sample.metainfo
        
        # Shape 정보를 올바른 형태로 강제 설정
        def ensure_shape_tuple(shape_value, default_shape):
            """Shape 값을 올바른 (H, W) tuple로 변환"""
            if shape_value is None:
                return default_shape
            
            # torch.Size, list, tuple 등을 처리
            if hasattr(shape_value, '__len__'):
                if len(shape_value) >= 2:
                    # 마지막 2개 차원을 H, W로 사용
                    h, w = int(shape_value[0]), int(shape_value[1])
                    return (h, w)
                elif len(shape_value) == 1:
                    # 정사각형으로 가정
                    size = int(shape_value[0])
                    return (size, size)
            
            # 단일 값인 경우 정사각형으로 처리
            if isinstance(shape_value, (int, float)):
                size = int(shape_value)
                return (size, size)
            
            # 기본값 반환
            return default_shape
        
        # 올바른 shape 설정
        ori_shape = ensure_shape_tuple(metainfo.get('ori_shape'), original_shape)
        img_shape = ensure_shape_tuple(metainfo.get('img_shape'), original_shape)
        pad_shape = final_shape  # final_shape 사용
        
        # 메타정보 업데이트
        metainfo.update({
            'ori_shape': ori_shape,      # (H, W) tuple
            'img_shape': img_shape,      # (H, W) tuple  
            'pad_shape': pad_shape,      # (H, W) tuple
            'scale_factor': (1.0, 1.0), # (scale_w, scale_h) tuple
            'flip': False,
            'flip_direction': None,
            'pad_size_divisor': self.pad_size_divisor
        })
    
    def _move_gt_to_device(self, data_sample, target_device):
        """GT 데이터를 타겟 디바이스로 이동"""
        
        if hasattr(data_sample, 'gt_instances') and data_sample.gt_instances is not None:
            gt_instances = data_sample.gt_instances
            # GT bboxes 이동
            if hasattr(gt_instances, 'bboxes') and gt_instances.bboxes is not None:
                gt_instances.bboxes = gt_instances.bboxes.to(target_device)
            # GT labels 이동
            if hasattr(gt_instances, 'labels') and gt_instances.labels is not None:
                gt_instances.labels = gt_instances.labels.to(target_device)
            # Ignore flags 이동 (있는 경우)
            if hasattr(gt_instances, 'ignore_flags') and gt_instances.ignore_flags is not None:
                gt_instances.ignore_flags = gt_instances.ignore_flags.to(target_device)
    
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


@MODELS.register_module() 
class DELIVERDataPreprocessorSimple(nn.Module):
    """Simplified DELIVER Data Preprocessor with metadata handling"""
    
    def __init__(self,
                 mean: List[List[float]] = None,
                 std: List[List[float]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: float = 0.0):
        super().__init__()
        
        if mean is None:
            mean = [
                [0.485, 0.456, 0.406],  # RGB
                [0.0, 0.0, 0.0],        # Depth
                [0.0, 0.0, 0.0],        # Event
                [0.0, 0.0, 0.0]         # LiDAR
            ]
            
        if std is None:
            std = [
                [0.229, 0.224, 0.225],  # RGB
                [1.0, 1.0, 1.0],        # Depth
                [1.0, 1.0, 1.0],        # Event
                [1.0, 1.0, 1.0]         # LiDAR
            ]
        
        self.mean_list = mean
        self.std_list = std
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        
        # Dummy parameter to track device
        self.register_parameter('_dummy', nn.Parameter(torch.empty(0)))
    
    def forward(self, data: Dict, training: bool = False) -> Dict:
        """Preprocess multimodal data with automatic device handling and metadata."""
        inputs = data['inputs']
        data_samples = data.get('data_samples', [])
        
        # Get target device from model
        target_device = self._dummy.device
        
        # Convert batch-wise tuples to modality-wise tensors if needed
        if isinstance(inputs[0], (tuple, list)):
            num_modalities = len(inputs[0])
            restructured_inputs = []
            
            for modal_idx in range(num_modalities):
                modal_batch = [batch_tuple[modal_idx] for batch_tuple in inputs]
                stacked_modal = torch.stack(modal_batch, dim=0)
                restructured_inputs.append(stacked_modal)
            
            inputs = restructured_inputs
        
        # Store original shape
        original_shape = inputs[0].shape[2:]  # (H, W)
        
        # Normalize each modality
        normalized_inputs = []
        
        for i, modal_tensor in enumerate(inputs):
            # Move to target device and convert to float
            modal_tensor = modal_tensor.float().to(target_device)
            
            # Create mean and std tensors on the same device as input
            mean = torch.tensor(self.mean_list[i], dtype=torch.float32, device=target_device).view(1, -1, 1, 1)
            std = torch.tensor(self.std_list[i], dtype=torch.float32, device=target_device).view(1, -1, 1, 1)
            
            # Normalize
            normalized_modal = (modal_tensor - mean) / std
            normalized_inputs.append(normalized_modal)
        
        # Apply padding if needed
        if self.pad_size_divisor > 1:
            normalized_inputs = self._pad_inputs(normalized_inputs)
        
        # Update metadata
        final_shape = normalized_inputs[0].shape[2:]  # (H, W) after padding
        
        for data_sample in data_samples:
            if not hasattr(data_sample, 'metainfo') or data_sample.metainfo is None:
                data_sample.set_metainfo({})
            
            metainfo = data_sample.metainfo
            metainfo['ori_shape'] = original_shape
            metainfo['img_shape'] = original_shape
            metainfo['pad_shape'] = final_shape
            metainfo['scale_factor'] = (1.0, 1.0)
            metainfo['pad_size_divisor'] = self.pad_size_divisor
        
        data['inputs'] = normalized_inputs
        data['data_samples'] = data_samples
        
        return data
    
    def _pad_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Pad inputs to be divisible by pad_size_divisor."""
        padded_inputs = []
        
        for modal_tensor in inputs:
            B, C, H, W = modal_tensor.shape
            pad_h = (self.pad_size_divisor - H % self.pad_size_divisor) % self.pad_size_divisor
            pad_w = (self.pad_size_divisor - W % self.pad_size_divisor) % self.pad_size_divisor
            
            if pad_h > 0 or pad_w > 0:
                # Padding: (left, right, top, bottom) = (0, pad_w, 0, pad_h)
                modal_tensor = torch.nn.functional.pad(
                    modal_tensor, (0, pad_w, 0, pad_h), value=self.pad_value
                )
            
            padded_inputs.append(modal_tensor)
        
        return padded_inputs