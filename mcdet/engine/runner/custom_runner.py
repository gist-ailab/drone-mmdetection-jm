# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import is_model_wrapper
from mmengine.runner import ValLoop
from mmdet.registry import LOOPS, RUNNERS
from mmengine.runner import Runner

@RUNNERS.register_module()

class FLIR_CatRunner(Runner):
    """Custom runner for flir_cat_dtataset"""
    pass


