#  mcdet/models/backbones/__init__.py


from .cmnext import CMNextBackbone
from .stitchfusion import StitchFusionBackbone
from .custom_resnet import ATTResNet
from .dinov2 import DINOv2
# from .base_backbone import BaseBackbone


__all__ = ['CMNextBackbone', 'StitchFusionBackbone','DINOv2']