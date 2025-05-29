# /mcdet/models/data_preprocessors/list_preprocessor.py
import math

from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import ImgDataPreprocessor
from mmengine.model.utils import stack_batch
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of

from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample




# /mcdet/models/data_preprocessors/list_preprocessor.py
import math

from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import ImgDataPreprocessor
from mmengine.model.utils import stack_batch
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of

from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample

@MODELS.register_module()
class ListDataPreprocessor(ImgDataPreprocessor):
    """Data preprocessor for list of multimodal images - DETR Compatible.

    Args:
        mean (Sequence[Number]): Mean values for each channel.
        std (Sequence[Number]): Standard deviation values for each channel.
        to_rgb (bool): Whether to convert images to RGB format.
        pad_size_divisor (int): The divisor to pad the image size.
        bgr_to_rgb (bool): Whether to convert BGR images to RGB format.
        bbox_format (str): Input bbox format ('xyxy' or 'xywh'). Default: 'xywh' (COCO format)
        pad_mode (str): Padding mode for different sized images. Options: 'stack', 'pad_to_max', 'pad_to_fixed'
        fixed_size (tuple): Fixed size for padding when pad_mode='pad_to_fixed'
    """

    def __init__(self,
                 mean_: List[List[float]] = None,
                 std_: List[List[float]] = None,
                 to_rgb: bool = True,
                 pad_size_divisor: Optional[int] = None,
                 bgr_to_rgb: bool = False,
                 bbox_format: str = 'xywh',
                 pad_mode: str = 'stack',
                 fixed_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(mean=None, std=None)
        self.pad_size_divisor = pad_size_divisor
        self.bgr_to_rgb = bgr_to_rgb
        self.bbox_format = bbox_format.lower()
        self.pad_mode = pad_mode
        self.fixed_size = fixed_size
        
        if mean_ is None:
            self.mean = [
                [0.485, 0.456, 0.406],       # RGB
                [0.0, 0.0, 0.0],             # Depth
                [0.0, 0.0, 0.0],             # Event
                [0.0, 0.0, 0.0]              # LiDAR
            ]
        else:
            self.mean = mean_
            
        if std_ is None:
            self.std = [
                [0.229, 0.224, 0.225],       # RGB
                [1.0, 1.0, 1.0],             # Depth
                [1.0, 1.0, 1.0],             # Event
                [1.0, 1.0, 1.0]              # LiDAR
            ]
        else:
            self.std = std_
            
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

    def _get_max_shape(self, modal_data: List[torch.Tensor]) -> Tuple[int, int]:
        """Get maximum height and width from all images in the batch."""
        max_h = max(img.shape[-2] for img in modal_data)
        max_w = max(img.shape[-1] for img in modal_data)
        return max_h, max_w

    def _pad_to_target_size(self, img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Pad single image to target size."""
        _, _, h, w = img.shape if img.dim() == 4 else (1, *img.shape)
        
        # Calculate padding
        pad_h = target_h - h
        pad_w = target_w - w
        
        if pad_h > 0 or pad_w > 0:
            # Pad: (pad_left, pad_right, pad_top, pad_bottom)
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
        
        return img

    def _batch_process(self, data: list):
        """Process batch with different sized images - DETR Compatible."""
        num_modals = len(data)
        batched_list = []
        
        if num_modals == 0:
            raise ValueError("Input data list is empty.")
        
        for modal_idx in range(num_modals):
            modal_data = data[modal_idx]
            
            # Convert to list if needed and move to device
            modal_tensors = []
            for img in modal_data:
                img_tensor = img.to(self.device)
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                modal_tensors.append(img_tensor)
            
            if self.pad_mode == 'stack':
                # Original behavior - requires same size
                try:
                    batched_modal = torch.stack(modal_tensors, dim=0)
                except RuntimeError as e:
                    raise RuntimeError(f"Cannot stack modal {modal_idx} images with different sizes. "
                                     f"Use pad_mode='pad_to_max' or 'pad_to_fixed'. Error: {e}")
            
            elif self.pad_mode == 'pad_to_max':
                # Pad all images to maximum size in the batch
                max_h, max_w = self._get_max_shape(modal_tensors)
                
                # Apply pad_size_divisor if specified
                if self.pad_size_divisor and self.pad_size_divisor > 1:
                    max_h = math.ceil(max_h / self.pad_size_divisor) * self.pad_size_divisor
                    max_w = math.ceil(max_w / self.pad_size_divisor) * self.pad_size_divisor
                
                padded_tensors = []
                for img_tensor in modal_tensors:
                    padded_img = self._pad_to_target_size(img_tensor, max_h, max_w)
                    padded_tensors.append(padded_img)
                
                batched_modal = torch.stack(padded_tensors, dim=0)
                # Remove the extra batch dimension from unsqueeze
                batched_modal = batched_modal.squeeze(1)
            
            elif self.pad_mode == 'pad_to_fixed':
                # Pad all images to fixed size
                if self.fixed_size is None:
                    raise ValueError("fixed_size must be specified when pad_mode='pad_to_fixed'")
                
                target_h, target_w = self.fixed_size
                
                # Apply pad_size_divisor if specified
                if self.pad_size_divisor and self.pad_size_divisor > 1:
                    target_h = math.ceil(target_h / self.pad_size_divisor) * self.pad_size_divisor
                    target_w = math.ceil(target_w / self.pad_size_divisor) * self.pad_size_divisor
                
                padded_tensors = []
                for img_tensor in modal_tensors:
                    padded_img = self._pad_to_target_size(img_tensor, target_h, target_w)
                    padded_tensors.append(padded_img)
                
                batched_modal = torch.stack(padded_tensors, dim=0)
                # Remove the extra batch dimension from unsqueeze
                batched_modal = batched_modal.squeeze(1)
            
            else:
                raise ValueError(f"Unknown pad_mode: {self.pad_mode}")
            
            batched_list.append(batched_modal)
        
        return batched_list

    @property
    def device(self) -> torch.device:
        """Get the device of the preprocessor."""
        return next(self.buffers()).device
    
    def _normalize(self, inputs: list) -> list:
        normalized_list = []
        for i in range(len(inputs)):
            modal_tensor = inputs[i]
            if i == 0:
                if self.bgr_to_rgb:
                    modal_tensor = modal_tensor[:, [2, 1, 0], :, :]
                modal_tensor = modal_tensor / 255.0
                mean = getattr(self, f'mean_{i}')
                std = getattr(self, f'std_{i}')
                normalized_tensor = (modal_tensor - mean) / std
            else:
                # For depth, event, and LiDAR, we assume they are already in the correct range
                normalized_tensor = modal_tensor / 255.0
                mean = getattr(self, f'mean_{i}')
                std = getattr(self, f'std_{i}')
                normalized_tensor = (normalized_tensor - mean) / std
            normalized_list.append(normalized_tensor)
        return normalized_list
        
    def _pad_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Additional padding if needed (for pad_size_divisor)."""
        if self.pad_mode in ['pad_to_max', 'pad_to_fixed']:
            # Padding is already handled in _batch_process
            return inputs
        
        padded_inputs = []
        for modal_tensor in inputs:
            B, C, H, W = modal_tensor.shape
            # Calculate padding
            pad_h = (self.pad_size_divisor - H % self.pad_size_divisor) % self.pad_size_divisor
            pad_w = (self.pad_size_divisor - W % self.pad_size_divisor) % self.pad_size_divisor
            
            if pad_h > 0 or pad_w > 0:
                modal_tensor = torch.nn.functional.pad(
                    modal_tensor, (0, pad_w, 0, pad_h), value=0.0
                )
            padded_inputs.append(modal_tensor)
        return padded_inputs

    def _convert_bbox_format(self, bboxes: torch.Tensor, src_format: str, dst_format: str = 'xyxy') -> torch.Tensor:
        """Convert bbox format between different coordinate systems."""
        if len(bboxes) == 0 or src_format == dst_format:
            return bboxes
            
        if src_format == 'xywh' and dst_format == 'xyxy':
            # COCO format (x, y, w, h) → xyxy format (x1, y1, x2, y2)
            x1 = bboxes[:, 0]                    # x (left)
            y1 = bboxes[:, 1]                    # y (top)
            x2 = bboxes[:, 0] + bboxes[:, 2]     # x + width (right)
            y2 = bboxes[:, 1] + bboxes[:, 3]     # y + height (bottom)
            return torch.stack([x1, y1, x2, y2], dim=1)
            
        elif src_format == 'xyxy' and dst_format == 'xywh':
            # xyxy format (x1, y1, x2, y2) → COCO format (x, y, w, h)
            x = bboxes[:, 0]                     # x1 (left)
            y = bboxes[:, 1]                     # y1 (top)
            w = bboxes[:, 2] - bboxes[:, 0]      # x2 - x1 (width)
            h = bboxes[:, 3] - bboxes[:, 1]      # y2 - y1 (height)
            return torch.stack([x, y, w, h], dim=1)
        else:
            raise ValueError(f"Unsupported format conversion: {src_format} -> {dst_format}")

    def _detect_bbox_format(self, bboxes: torch.Tensor) -> str:
        """Automatically detect bbox format."""
        if len(bboxes) == 0:
            return 'xyxy'  # Default
            
        # Check if x2 > x1 and y2 > y1 for all bboxes
        x2_gt_x1 = (bboxes[:, 2] > bboxes[:, 0]).all()
        y2_gt_y1 = (bboxes[:, 3] > bboxes[:, 1]).all()
        
        if x2_gt_x1 and y2_gt_y1:
            return 'xyxy'
        else:
            return 'xywh'

    def _convert_gt_bboxes_to_xyxy(self, data_samples: List):
        """Convert GT bboxes to xyxy format."""
        if not data_samples:
            return
            
        for i, data_sample in enumerate(data_samples):
            if not hasattr(data_sample, 'gt_instances') or data_sample.gt_instances is None:
                continue
                
            gt_instances = data_sample.gt_instances
            if not hasattr(gt_instances, 'bboxes') or gt_instances.bboxes is None:
                continue
                
            if len(gt_instances.bboxes) == 0:
                continue
                
            # Ensure bboxes is a tensor
            if not torch.is_tensor(gt_instances.bboxes):
                gt_instances.bboxes = torch.tensor(gt_instances.bboxes, dtype=torch.float32)
            
            # Auto-detect format if not specified
            if hasattr(self, 'bbox_format') and self.bbox_format:
                detected_format = self.bbox_format
            else:
                detected_format = self._detect_bbox_format(gt_instances.bboxes)
            
            # Convert to xyxy format
            if detected_format != 'xyxy':
                gt_instances.bboxes = self._convert_bbox_format(
                    gt_instances.bboxes, 
                    src_format=detected_format, 
                    dst_format='xyxy'
                )
            
            # Validate converted bboxes
            self._validate_bboxes(gt_instances.bboxes, data_sample, i)

    def _validate_bboxes(self, bboxes: torch.Tensor, data_sample, sample_idx: int):
        """Validate bbox coordinates."""
        if len(bboxes) == 0:
            return
            
        # Check for negative coordinates
        if (bboxes < 0).any():
            print(f"Warning: Negative coordinates found in sample {sample_idx}")
            
        # Check for invalid boxes (x2 <= x1 or y2 <= y1)
        invalid_width = (bboxes[:, 2] <= bboxes[:, 0]).any()
        invalid_height = (bboxes[:, 3] <= bboxes[:, 1]).any()
        
        if invalid_width or invalid_height:
            print(f"Warning: Invalid bbox dimensions in sample {sample_idx}")
            print(f"  Bboxes: {bboxes}")

    def _move_gt_to_device(self, data_samples: List, target_device: torch.device):
        """Move ground truth data to target device."""
        if not data_samples:
            return
            
        for data_sample in data_samples:
            if hasattr(data_sample, 'gt_instances') and data_sample.gt_instances is not None:
                gt_instances = data_sample.gt_instances
                
                # Move GT bboxes to target device
                if hasattr(gt_instances, 'bboxes') and gt_instances.bboxes is not None:
                    if torch.is_tensor(gt_instances.bboxes):
                        gt_instances.bboxes = gt_instances.bboxes.to(target_device)
                
                # Move GT labels to target device
                if hasattr(gt_instances, 'labels') and gt_instances.labels is not None:
                    if torch.is_tensor(gt_instances.labels):
                        gt_instances.labels = gt_instances.labels.to(target_device)
                
                # Move ignore flags to target device (if present)
                if hasattr(gt_instances, 'ignore_flags') and gt_instances.ignore_flags is not None:
                    if torch.is_tensor(gt_instances.ignore_flags):
                        gt_instances.ignore_flags = gt_instances.ignore_flags.to(target_device)

    def _update_data_sample_metainfo(self, data_samples: List, padded_inputs: List[torch.Tensor]):
        """Update data sample metainfo with proper shape information."""
        if not data_samples or not padded_inputs:
            return
            
        # Get shape information from first modality (assumed to be primary)
        padded_shape = padded_inputs[0].shape[2:]  # (H, W)
        batch_input_shape = padded_shape
        
        for data_sample in data_samples:
            if hasattr(data_sample, 'set_metainfo'):
                # Get existing metainfo or create new dict
                if hasattr(data_sample, 'metainfo') and data_sample.metainfo is not None:
                    current_metainfo = dict(data_sample.metainfo)  # Make a copy
                else:
                    current_metainfo = {}
                
                # Update metainfo with new values
                current_metainfo.update({
                    'pad_shape': padded_shape,
                    'batch_input_shape': batch_input_shape,
                    'pad_size_divisor': self.pad_size_divisor or 1,
                    'pad_mode': self.pad_mode,
                })
                
                # Set the updated metainfo back
                data_sample.set_metainfo(current_metainfo)
                
            else:
                if not hasattr(data_sample, 'metainfo') or data_sample.metainfo is None:
                    data_sample.metainfo = {}
                
                # Direct assignment approach
                data_sample.metainfo['pad_shape'] = padded_shape
                data_sample.metainfo['batch_input_shape'] = batch_input_shape  
                data_sample.metainfo['pad_size_divisor'] = self.pad_size_divisor or 1
                data_sample.metainfo['pad_mode'] = self.pad_mode

    def forward(self, data: dict, training: bool = False) -> dict:
        """
        Perform multimodal data preprocessing, normalization, and padding for DETR.
        """
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)

        # Convert GT bboxes to xyxy format BEFORE any other processing
        if data_samples and training:
            self._convert_gt_bboxes_to_xyxy(data_samples)

        # Process batch with different sized images
        _batch_inputs = self._batch_process(inputs)
        _batch_inputs = self._normalize(_batch_inputs)
        _padded_inputs = self._pad_inputs(_batch_inputs)
        
        # Move GT data to device and update metainfo
        if data_samples:
            self._move_gt_to_device(data_samples, self.device)
            self._update_data_sample_metainfo(data_samples, _padded_inputs)
                    
        # Return processed data
        return {
            'inputs': _padded_inputs,
            'data_samples': data_samples
        }