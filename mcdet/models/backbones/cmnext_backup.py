# mcdet/models/backbones/cmnext.py

import torch
import math
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Union, Optional, Tuple

from mmdet.registry import MODELS
from mmdet.models.backbones.base_backbone import BaseBackbone
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader

# Import your semseg components (adjust path as needed)
try:
    from semseg.models.backbones import *
    from semseg.models.layers import trunc_normal_
except ImportError:
    print("Warning: semseg not found. Make sure to install semseg library.")
    # Fallback implementation or raise error
    pass


def load_dualpath_model(model, model_file):
    """Load pretrained model for multimodal CMNext."""
    # load raw state_dict
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[CMNext] Pretrained model loaded: {msg}")
    del state_dict


class CMNextBaseModel(BaseModule):
    """Base model wrapper for CMNext backbone."""
    
    def __init__(self, 
                 backbone: str = 'MiT-B0', 
                 modals: List[str] = ['rgb', 'depth', 'event', 'lidar'],
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        
        backbone_name, variant = backbone.split('-')
        self.backbone = eval(backbone_name)(variant, modals)
        self.modals = modals

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        """Load pretrained weights."""
        if pretrained:
            if len(self.modals) > 1:
                load_dualpath_model(self.backbone, pretrained)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print(f"[CMNext] Single modal pretrained loaded: {msg}")


@MODELS.register_module()
class CMNextBackbone(BaseBackbone):
    """CMNext backbone for multimodal object detection.
    
    This backbone processes multimodal inputs (RGB, Depth, Event, LiDAR) 
    and outputs multi-scale features for detection.
    
    Args:
        backbone (str): Backbone variant, e.g., 'CMNeXt-B0', 'CMNeXt-B2'
        modals (list): List of modalities to process
        out_indices (tuple): Output indices for FPN
        frozen_stages (int): Stages to be frozen
        norm_eval (bool): Whether to set norm layers to eval mode
        pretrained (str): Path to pretrained weights
    """
    
    def __init__(self,
                 backbone: str = 'CMNeXt-B2',
                 modals: List[str] = ['rgb', 'depth', 'event', 'lidar'],
                 out_indices: Tuple[int] = (0, 1, 2, 3),
                 frozen_stages: int = -1,
                 norm_eval: bool = False,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[dict] = None):
        
        super().__init__(init_cfg=init_cfg)
        
        self.backbone_name = backbone
        self.modals = modals
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        
        # Create CMNext base model
        self.cmnext_model = CMNextBaseModel(
            backbone=backbone, 
            modals=modals,
            init_cfg=init_cfg
        )
        
        # Determine output channels based on backbone variant
        if 'B0' in backbone or 'B1' in backbone:
            self.out_channels = [64, 128, 320, 512]  # Typical MiT-B0/B1 channels
        elif 'B2' in backbone:
            self.out_channels = [64, 128, 320, 512]  # MiT-B2 channels
        elif 'B3' in backbone:
            self.out_channels = [64, 128, 320, 512]  # MiT-B3 channels
        elif 'B4' in backbone:
            self.out_channels = [64, 128, 320, 512]  # MiT-B4 channels
        elif 'B5' in backbone:
            self.out_channels = [64, 128, 320, 512]  # MiT-B5 channels
        else:
            self.out_channels = [64, 128, 320, 512]  # Default
        
        # Load pretrained weights if provided
        if pretrained:
            self.cmnext_model.init_pretrained(pretrained)
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze stages according to frozen_stages."""
        if self.frozen_stages >= 0:
            # Freeze patch embedding
            if hasattr(self.cmnext_model.backbone, 'patch_embed'):
                for param in self.cmnext_model.backbone.patch_embed.parameters():
                    param.requires_grad = False
            
            # Freeze stages
            for i in range(self.frozen_stages + 1):
                if hasattr(self.cmnext_model.backbone, f'block{i+1}'):
                    for param in getattr(self.cmnext_model.backbone, f'block{i+1}').parameters():
                        param.requires_grad = False
    
    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Forward pass of CMNext backbone.
        
        Args:
            x: List of multimodal tensors [rgb_tensor, depth_tensor, event_tensor, lidar_tensor]
               Each tensor has shape (B, C, H, W)
        
        Returns:
            Tuple of feature tensors from different stages
        """
        # CMNext backbone expects list of multimodal inputs
        features = self.cmnext_model.backbone(x)
        
        # Convert to MMDetection standard format
        # features should be a list/dict with multi-scale outputs
        if isinstance(features, dict):
            # If features is dict with keys like '0', '1', '2', '3'
            outs = []
            for idx in self.out_indices:
                if str(idx) in features:
                    outs.append(features[str(idx)])
                elif idx in features:
                    outs.append(features[idx])
        elif isinstance(features, (list, tuple)):
            # If features is list/tuple
            outs = [features[i] for i in self.out_indices]
        else:
            raise ValueError(f"Unexpected features format: {type(features)}")
        
        return tuple(outs)
    
    def train(self, mode: bool = True):
        """Set train/eval mode."""
        super().train(mode)
        
        if mode and self.norm_eval:
            # Set norm layers to eval mode
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    m.eval()
    
    def init_weights(self):
        """Initialize weights of the backbone."""
        if self.init_cfg is None:
            # Apply custom weight initialization
            self.cmnext_model.apply(self.cmnext_model._init_weights)
        else:
            super().init_weights()


@MODELS.register_module()
class CMNextBackboneWithFPN(BaseBackbone):
    """CMNext backbone with built-in FPN for detection.
    
    This combines CMNext backbone with Feature Pyramid Network
    for better multi-scale feature fusion.
    """
    
    def __init__(self,
                 backbone: str = 'CMNeXt-B2',
                 modals: List[str] = ['rgb', 'depth', 'event', 'lidar'],
                 out_indices: Tuple[int] = (0, 1, 2, 3),
                 fpn_in_channels: List[int] = [64, 128, 320, 512],
                 fpn_out_channels: int = 256,
                 fpn_num_outs: int = 5,
                 frozen_stages: int = -1,
                 norm_eval: bool = False,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[dict] = None):
        
        super().__init__(init_cfg=init_cfg)
        
        # CMNext backbone
        self.backbone = CMNextBackbone(
            backbone=backbone,
            modals=modals,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            norm_eval=norm_eval,
            pretrained=pretrained,
            init_cfg=init_cfg
        )
        
        # Feature Pyramid Network
        self.fpn = FPN(
            in_channels=fpn_in_channels,
            out_channels=fpn_out_channels,
            num_outs=fpn_num_outs,
            start_level=0,
            end_level=-1,
            add_extra_convs='on_output'
        )
        
        self.out_channels = fpn_out_channels
    
    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Forward pass with FPN."""
        # Get backbone features
        backbone_features = self.backbone(x)
        
        # Apply FPN
        fpn_features = self.fpn(backbone_features)
        
        return fpn_features


# Simple FPN implementation (you can replace with mmdet's FPN)
class FPN(BaseModule):
    """Simple Feature Pyramid Network."""
    
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: str = 'on_output',
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.num_ins):
            l_conv = nn.Conv2d(in_channels[i], out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        # Extra convolutions for additional levels
        if add_extra_convs == 'on_output':
            self.extra_convs = nn.ModuleList()
            for i in range(num_outs - self.num_ins):
                extra_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                self.extra_convs.append(extra_conv)
    
    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Forward pass of FPN."""
        # Lateral connections
        laterals = [conv(inputs[i]) for i, conv in enumerate(self.lateral_convs)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + nn.functional.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode='nearest'
            )
        
        # FPN outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        
        # Extra outputs
        if hasattr(self, 'extra_convs'):
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[i](outs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        
        return tuple(outs)