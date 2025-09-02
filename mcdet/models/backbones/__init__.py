# # mcdet/models/backbones/__init__.py

# mcdet/models/backbones/__init__.py
from .cmnext import CMNextBackbone
from .stitchfusion import StitchFusionBackbone
from .custom_resnet import ATTResNet
from .geminifusion import GeminiFusionBackbone
from .cmnextp import CMNeXtPBackbone

# from .base_backbone import BaseBackbone


__all__ = ['CMNextBackbone', 'StitchFusionBackbone', 'GeminiFusionBackbone', 'CMNeXtPBackbone']