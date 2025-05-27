# # mcdet/models/backbones/__init__.py
# from .cmnext import CMNextBackbone, CMNextBackboneWithFPN
# from .custom_resnet import ATTResNet
# # from .base_backbone import BaseBackbone


# __all__ = ['CMNextBackbone', 'CMNextBackboneWithFPN', 'ATTResNet', 'BaseBackbone']

# mcdet/models/backbones/__init__.py
from .cmnext import CMNextBackbone
from .custom_resnet import ATTResNet
# from .base_backbone import BaseBackbone


__all__ = ['CMNextBackbone']