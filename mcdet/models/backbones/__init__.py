# mcdet/models/backbones/__init__.py

# 각 커스텀 백본 파이썬 파일로부터 클래스를 임포트합니다.
from .cmnext import CMNextBackbone
from .stitchfusion import StitchFusionBackbone
from .custom_resnet import ATTResNet
from .geminifusion import GeminiFusionBackbone
from .cmnextp import CMNeXtPBackbone
from .cmnextpsp import CMNeXtPSPBackbone

__all__ = [
    'CMNextBackbone', 'StitchFusionBackbone', 'ATTResNet',
    'GeminiFusionBackbone', 'CMNeXtPBackbone', 'CMNeXtPSPBackbone'
]