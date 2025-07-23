
import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from semseg.models.layers import DropPath
import torch.nn.init as init
import numpy as np
from math import factorial
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from mcdet.models.modules.ffm import FeatureFusionModule as FFM
from mcdet.models.modules.mspa import MSPABlock
from mcdet.models.modules.ffm import FeatureRectifyModule as FRM
import functools
from functools import partial
import warnings
import torch.nn.functional as F
import math
from typing import Dict, List, Union, Optional, Tuple


from mcdet.models.backbones.cmnext import MLP, DWConv, Attention, PatchEmbed

@MODELS.register_module() 
class DINOv2(BaseModule):
    def __init__(self,
                 arch='b',                          
                 img_size=448,
                 patch_size=14,                     
                 out_indices=(0, 1, 2, 3),
                 norm_eval=True,
                 frozen_stages=-1,  # -1: freeze 안함, 0~11: 해당 stage까지 freeze, 12: 모든 블록 freeze
                 init_cfg=None):
        super().__init__(init_cfg)
        
        # DINOv2 모델 로드
        self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{arch}{patch_size}')
        
        self.out_indices = out_indices
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Feature map 크기 계산
        self.patch_embed = self.model.patch_embed
        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        # frozen_stages 적용
        self._freeze_stages()
        
    def _freeze_stages(self):
        """Freeze stages according to frozen_stages.
            -1: unfreeze
            0: freeze patch embedding only
            1: patch_embed + freeze first block
            12: 모든 블록 freeze (DINOv2-base는 12개 블록)
            999: 전체 모델 freeze
        """

        if self.frozen_stages >= 0:
            # Patch embedding freeze
            for param in self.model.patch_embed.parameters():
                param.requires_grad = False
            
            # Position embedding freeze (if exists)
            if hasattr(self.model, 'pos_embed'):
                self.model.pos_embed.requires_grad = False
            
            # Class token freeze (if exists)
            if hasattr(self.model, 'cls_token'):
                self.model.cls_token.requires_grad = False
            
            # Blocks freeze
            for i in range(min(self.frozen_stages, len(self.model.blocks))):
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = False
            
            # 모든 블록을 freeze하려면
            if self.frozen_stages >= len(self.model.blocks):
                for param in self.model.parameters():
                    param.requires_grad = False
                    
    def forward(self, x):
        # DINOv2 forward pass
        features = []
        
        x1 = self.model.prepare_tokens_with_masks(x, None)
        
        for i, blk in enumerate(self.model.blocks):
            x1 = blk(x1)  # x1을 업데이트하도록 수정
            if i in self.out_indices:
                # [B, N+1, C] -> [B, H, W, C] -> [B, C, H, W]
                cls_token, patch_tokens = x1[:, 0], x1[:, 1:]
                B, N, C = patch_tokens.shape
                
                # 미리 계산된 크기 사용
                H = W = self.num_patches_per_side
                
                # 실제 N이 예상과 다른 경우 디버깅 정보 출력
                if N != self.num_patches:
                    print(f"Warning: Expected {self.num_patches} patches, got {N}")
                    H = W = int(math.sqrt(N))
                
                patch_tokens = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
                features.append(patch_tokens)
        
        return tuple(features)

    def train(self, mode=True):
        """Set train/eval mode."""
        super().train(mode)
        
        if mode and self.norm_eval:
            # Set norm layers to eval mode
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    m.eval()