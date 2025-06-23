# # mcdet/models/backbones/__init__.py
# from .cmnext import CMNextBackbone, CMNextBackboneWithFPN
# from .custom_resnet import ATTResNet
# # from .base_backbone import BaseBackbone


# __all__ = ['CMNextBackbone', 'CMNextBackboneWithFPN', 'ATTResNet', 'BaseBackbone']

# mcdet/models/backbones/__init__.py
from .cmnext import CMNextBackbone
from .stitchfusion import StitchFusionBackbone
from .custom_resnet import ATTResNet
from .geminifusion import GeminiFusion
from .geminifusion_second import GeminiFusion_second
# from .base_backbone import BaseBackbone


__all__ = ['CMNextBackbone', 'StitchFusionBackbone', 'GeminiFusion', 'ATTResNet', 'GeminiFusion_second']