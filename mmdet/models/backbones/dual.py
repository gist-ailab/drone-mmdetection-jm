# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
# from .backbone import * 
from .backbone import ResNet
import torch

from mmdet.registry import MODELS
from ..layers import ResLayer


class DualBackbone(BaseModule):
    def __init__(self, backbone_cfg, init_cfg=None):
        super().__init__(init_cfg)
        self.rgb_backbone = ResNet(**backbone_cfg)
        self.ir_backbone = ResNet(**backbone_cfg)  # Clone of RGB backbone for IR images

    def forward(self, rgb_img, ir_img):
        rgb_feats = self.rgb_backbone(rgb_img)
        ir_feats = self.ir_backbone(ir_img)

        # Concatenate features from both backbones
        combined_feats = [torch.cat((rgb_feat, ir_feat), dim=1) for rgb_feat, ir_feat in zip(rgb_feats, ir_feats)]
        return combined_feats