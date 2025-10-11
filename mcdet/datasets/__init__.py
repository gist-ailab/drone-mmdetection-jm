# mcdet/datasets/__init__.py

# 1. 모든 데이터셋 클래스들을 임포트합니다.
from .custom_flir_dataset import FLIRCatDataset, FLIRCatDataset2
from .custom_drone_Dataset import GISTDataset
from .kaist_rgbt_coco_dataset import KaistRgbtCocoDataset
from .llvip_dataset import LLVIPRgbtDataset
from .flir_aligned_coco_dataset import FLIRRgbtCocoDataset
from .custom_deliver_detection_dataset import DELIVERDetectionDataset
from .custom_sampler import CustomSampler
from .custom_sejong_detection_dataset import SejongDetectionDataset

# 2. transforms 서브모듈을 임포트하여 모든 커스텀 transform을 등록합니다.
from .transforms import * # <--- 이 줄을 추가하세요!

# 3. __all__을 정의할 때, 데이터셋 클래스들과 transforms의 클래스들을 모두 포함시킵니다.
__all__ = [
    'FLIRCatDataset', 'FLIRCatDataset2', 'GISTDataset', 'KaistRgbtCocoDataset',
    'FLIRRgbtCocoDataset', 'DELIVERDetectionDataset',
    'CustomSampler', 'SejongDetectionDataset'
] + transforms.__all__  