# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.models import build_detector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector


@MODELS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

@DATASETS.register_module()
class CustomFasterRCNN(FasterRCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Customize the backbone
        backbone_cfg = dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        )
        self.backbone = DualBackbone(backbone_cfg)

        # Customize the FPN
        self.neck = FPN(
            in_channels=fpn_in_channels,
            out_channels=256,
            num_outs=5
        )

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck"""
        if isinstance(img, tuple):
            rgb_img, ir_img = img
        else:
            raise TypeError("img must be a tuple (rgb_img, ir_img)")
        x = self.backbone(rgb_img, ir_img)
        x = self.neck(x)
        return x