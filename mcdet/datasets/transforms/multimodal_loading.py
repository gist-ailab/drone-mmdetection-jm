# mcdet/datasets/transforms/multimodal_loading.py
import mmcv
import numpy as np
# from mmdet.datasets.transforms import LoadImageFromFile
from mmcv.transforms import LoadImageFromFile

from mmdet.registry import TRANSFORMS
from typing import Dict, List, Optional, Union


@TRANSFORMS.register_module()
class LoadDELIVERImages(LoadImageFromFile):
    """Load DELIVER multimodal images (RGB, Depth, Event, LiDAR).
    
    This transform loads 4 modalities:
    - RGB: Standard color image
    - Depth: Depth information
    - Event: Event camera data 
    - LiDAR: LiDAR projected image
    
    All images are loaded and stored as a list in the format:
    [rgb_img, depth_img, event_img, lidar_img]
    """
    
    def __init__(self, 
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 **kwargs):
        super().__init__(
            to_float32=to_float32,
            color_type=color_type,
            imdecode_backend=imdecode_backend,
            file_client_args=file_client_args,
            ignore_empty=ignore_empty,
            **kwargs
        )
    
    def transform(self, results: Dict) -> Dict:
        """Load multimodal images.
        
        Args:
            results (dict): Result dict containing image paths
            
        Returns:
            dict: Results with loaded multimodal images
        """
        modality_images = []
        modality_paths = results['modality_paths']
        
        for modality in ['rgb', 'depth', 'event', 'lidar']:
            img_path = modality_paths[modality]
            
            # Load image using parent class method
            temp_results = {'img_path': img_path}
            temp_results = super().transform(temp_results)
            
            modality_images.append(temp_results['img'])
        
        # Store as list of images
        results['img'] = modality_images
        results['img_path'] = [modality_paths[mod] for mod in ['rgb', 'depth', 'event', 'lidar']]
        results['img_shape'] = modality_images[0].shape[:2]  
        results['ori_shape'] = results['img_shape']  

        return results