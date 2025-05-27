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
    """Data preprocessor for list of images.

    Args:
        mean (Sequence[Number]): Mean values for each channel.
        std (Sequence[Number]): Standard deviation values for each channel.
        to_rgb (bool): Whether to convert images to RGB format.
        pad_size_divisor (int): The divisor to pad the image size.
        bgr_to_rgb (bool): Whether to convert BGR images to RGB format.
    """

    def __init__(self,
                 mean_: List[List[float]] = None,
                 std_: List[List[float]] = None,
                 to_rgb: bool = True,
                 pad_size_divisor: Optional[int] = None,
                 bgr_to_rgb: bool = False) -> None:
        super().__init__(mean=None, std=None)
        self.pad_size_divisor = pad_size_divisor
        self.bgr_to_rgb = bgr_to_rgb
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

    def _batch_process(self, data: list):
        num_modals = len(data)
        batched_list = []
        if num_modals == 0:
            raise ValueError("Input data list is empty.")
        for idx in range(num_modals):
            modal_data = data[idx]
            batched_modal = torch.stack(list(modal_data), dim=0)
            batched_modal = batched_modal.to(self.device)
            batched_list.append(batched_modal)
        return batched_list

    @property
    def device(self) -> torch.device:
        """Get the device of the preprocessor."""
        return next(self.buffers()).device
    
    def _normalize(self, inputs: list) -> list:
        # rgb_tensors = inputs[0]
        normalized_list = []
        for i in range(len(inputs)):
            modal_tensor = inputs[i]
            if i == 0:
                if self.bgr_to_rgb:
                    inputs[i] = inputs[i][:, [2, 1, 0], :, :]
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
        """Pad inputs to be divisible by pad_size_divisor."""
        padded_inputs = []
        pad_shapes = []  
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
            pad_shapes.append((H + pad_h, W + pad_w))
        return padded_inputs

    def _move_gt_to_device(self, data_samples: List, target_device: torch.device):
        """Move ground truth data to target device."""
        if not data_samples:
            return
            
        for data_sample in data_samples:
            if hasattr(data_sample, 'gt_instances') and data_sample.gt_instances is not None:
                gt_instances = data_sample.gt_instances
                
                # 🔥 Move GT bboxes to target device
                if hasattr(gt_instances, 'bboxes') and gt_instances.bboxes is not None:
                    if torch.is_tensor(gt_instances.bboxes):
                        gt_instances.bboxes = gt_instances.bboxes.to(target_device)
                
                # 🔥 Move GT labels to target device
                if hasattr(gt_instances, 'labels') and gt_instances.labels is not None:
                    if torch.is_tensor(gt_instances.labels):
                        gt_instances.labels = gt_instances.labels.to(target_device)
                
                # 🔥 Move ignore flags to target device (if present)
                if hasattr(gt_instances, 'ignore_flags') and gt_instances.ignore_flags is not None:
                    if torch.is_tensor(gt_instances.ignore_flags):
                        gt_instances.ignore_flags = gt_instances.ignore_flags.to(target_device)


    def forward(self, data: dict, training: bool = False) -> dict:
        """
        Perform list data preprocess, normalization, padding 
        """
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)

        _batch_inputs = self._batch_process(inputs)
        _batch_inputs = self._normalize(_batch_inputs)
        _padded_inputs = self._pad_inputs(_batch_inputs)
        batch_input_shape = [_padded_inputs[i].shape for i in range(len(_padded_inputs))]
        # print(f"  Final shapes: {batch_input_shape}")
        
        # Update metadata for data samples
        if data_samples:
            self._move_gt_to_device(data_samples, self.device)
            padded_shape = _padded_inputs[0].shape[2:]  # (H, W)
            for data_sample in data_samples:
                if not hasattr(data_sample, 'metainfo') or data_sample.metainfo is None:
                    data_sample.set_metainfo({})

                metainfo = data_sample.metainfo
                # Update shape information
                if 'pad_shape' not in metainfo:
                    metainfo['pad_shape'] = padded_shape
                if 'batch_input_shape' not in metainfo:
                    metainfo['batch_input_shape'] = padded_shape
                if 'pad_size_divisor' not in metainfo:
                    metainfo['pad_size_divisor'] = self.pad_size_divisor or 1
        # Return processed data
        return {
            'inputs': _padded_inputs,
            'data_samples': data_samples
        }
        
