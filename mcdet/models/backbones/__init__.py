# mcdet/models/backbones/__init__.py
from .cmnext import CMNextBackbone, CMNextBackboneWithFPN
from .custom_resnet import ATTResNet

__all__ = ['CMNextBackbone', 'CMNextBackboneWithFPN', 'ATTResNet']