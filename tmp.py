# import sys
# import os
# sys.path.append('/media/jemo/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm')

# import numpy as np
# import json

# def simple_bbox_check():
#     """Simple bbox analysis by reading JSON files directly"""
    
#     # 직접 JSON 파일 읽기
#     ann_file = '/media/jemo/HDD1/Workspace/dset/DELIVER/coco_train_xywh.json'
    
#     if not os.path.exists(ann_file):
#         print(f"Annotation file not found: {ann_file}")
#         return
    
#     print(f"Reading {ann_file}...")
#     with open(ann_file, 'r') as f:
#         coco_data = json.load(f)
    
#     annotations = coco_data['annotations']
#     images = {img['id']: img for img in coco_data['images']}
    
#     print(f"Total annotations: {len(annotations)}")
#     print(f"Total images: {len(images)}")
    
#     # bbox 분석
#     widths, heights, areas = [], [], []
#     small_bboxes = []
    
#     for i, ann in enumerate(annotations):
#         if i >= 1000:  # 처음 1000개만 분석
#             break
            
#         bbox = ann['bbox']  # [x, y, w, h]
#         x, y, w, h = bbox
        
#         widths.append(w)
#         heights.append(h)
#         areas.append(w * h)
        
#         # 작은 bbox 기록
#         if w < 10 or h < 10:
#             img_info = images.get(ann['image_id'], {})
#             small_bboxes.append({
#                 'bbox': bbox,
#                 'area': ann['area'],
#                 'category_id': ann['category_id'],
#                 'image_id': ann['image_id'],
#                 'image_file': img_info.get('file_name', 'unknown')
#             })
    
#     if not widths:
#         print("No bboxes found!")
#         return
    
#     # 통계 출력
#     print(f"\n=== BBOX STATISTICS (first {len(widths)} annotations) ===")
#     print(f"Width stats: min={min(widths):.2f}, max={max(widths):.2f}, mean={np.mean(widths):.2f}")
#     print(f"Height stats: min={min(heights):.2f}, max={max(heights):.2f}, mean={np.mean(heights):.2f}")
#     print(f"Area stats: min={min(areas):.2f}, max={max(areas):.2f}, mean={np.mean(areas):.2f}")
    
#     # 작은 bbox 분석
#     small_width = sum(1 for w in widths if w < 10)
#     small_height = sum(1 for h in heights if h < 10)
#     tiny_width = sum(1 for w in widths if w < 5)
#     tiny_height = sum(1 for h in heights if h < 5)
    
#     print(f"\n=== SMALL BBOX ANALYSIS ===")
#     print(f"Width < 10: {small_width}/{len(widths)} ({small_width/len(widths)*100:.1f}%)")
#     print(f"Height < 10: {small_height}/{len(heights)} ({small_height/len(heights)*100:.1f}%)")
#     print(f"Width < 5: {tiny_width}/{len(widths)} ({tiny_width/len(widths)*100:.1f}%)")
#     print(f"Height < 5: {tiny_height}/{len(heights)} ({tiny_height/len(heights)*100:.1f}%)")
    
#     # 가장 작은 bbox들 출력
#     print(f"\n=== SMALLEST BBOXES (showing first 10) ===")
#     small_bboxes_sorted = sorted(small_bboxes, key=lambda x: x['bbox'][2] * x['bbox'][3])
#     for i, bbox_info in enumerate(small_bboxes_sorted[:10]):
#         x, y, w, h = bbox_info['bbox']
#         print(f"{i+1:2d}. Size: {w:.1f}x{h:.1f} (area={w*h:.1f}), "
#               f"Cat: {bbox_info['category_id']}, File: {bbox_info['image_file']}")
    
#     # 분포 분석
#     width_ranges = [
#         (0, 5, "tiny"),
#         (5, 10, "small"), 
#         (10, 20, "medium-small"),
#         (20, 50, "medium"),
#         (50, 100, "large"),
#         (100, float('inf'), "very large")
#     ]
    
#     print(f"\n=== WIDTH DISTRIBUTION ===")
#     for min_w, max_w, label in width_ranges:
#         if max_w == float('inf'):
#             count = sum(1 for w in widths if w >= min_w)
#         else:
#             count = sum(1 for w in widths if min_w <= w < max_w)
#         print(f"{label:12s} ({min_w:3.0f}-{max_w if max_w != float('inf') else '∞':>3s}): "
#               f"{count:4d} ({count/len(widths)*100:5.1f}%)")
    
#     print(f"\n=== HEIGHT DISTRIBUTION ===")
#     for min_h, max_h, label in width_ranges:  # 같은 범위 사용
#         if max_h == float('inf'):
#             count = sum(1 for h in heights if h >= min_h)
#         else:
#             count = sum(1 for h in heights if min_h <= h < max_h)
#         print(f"{label:12s} ({min_h:3.0f}-{max_h if max_h != float('inf') else '∞':>3s}): "
#               f"{count:4d} ({count/len(heights)*100:5.1f}%)")
    
#     return widths, heights, areas

# def check_dataset_loading():
#     """Check if dataset can be loaded with current pipeline"""
#     try:
#         from mmengine import Config
#         from mcdet.datasets.custom_deliver_detection_dataset import DELIVERDetectionDataset
        
#         cfg = Config.fromfile('custom_configs/DELIVER/deliver_cmnext_rcnn.py')
#         dataset_cfg = cfg.train_dataloader.dataset
        
#         print(f"Creating dataset with config:")
#         print(f"  data_root: {dataset_cfg.data_root}")
#         print(f"  ann_file: {dataset_cfg.ann_file}")
#         print(f"  pipeline length: {len(dataset_cfg.pipeline)}")
        
#         dataset = DELIVERDetectionDataset(
#             data_root=dataset_cfg.data_root,
#             ann_file=dataset_cfg.ann_file,
#             data_prefix=dataset_cfg.data_prefix,
#             pipeline=dataset_cfg.pipeline,
#             metainfo=dataset_cfg.metainfo,
#             filter_cfg=dataset_cfg.get('filter_cfg', None)
#         )
        
#         print(f"Dataset created successfully!")
#         print(f"Dataset length: {len(dataset)}")
        
#         if len(dataset) > 0:
#             print(f"\nTesting first sample...")
#             data = dataset[0]
#             print(f"Sample keys: {list(data.keys())}")
            
#             if 'data_samples' in data:
#                 gt_instances = data['data_samples'].gt_instances
#                 bboxes = gt_instances.bboxes
#                 labels = gt_instances.labels
                
#                 print(f"Number of objects: {len(bboxes)}")
#                 print(f"Bbox shapes: {bboxes.shape}")
#                 print(f"First few bboxes:")
#                 for i, bbox in enumerate(bboxes[:3]):
#                     x, y, w, h = bbox.tolist()
#                     print(f"  {i+1}. [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}] (label: {labels[i]})")
                
#                 return True
        
#     except Exception as e:
#         print(f"Dataset loading failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
    
#     return False

# if __name__ == "__main__":
#     print("=== SIMPLE BBOX ANALYSIS ===")
#     print("1. Checking dataset loading...")
#     dataset_ok = check_dataset_loading()
    
#     print(f"\n2. Analyzing bbox distribution from JSON...")
#     simple_bbox_check()
    
#     if not dataset_ok:
#         print(f"\n⚠️  Dataset loading failed. Check your config and pipeline!")
#     else:
#         print(f"\n✅ Dataset loading successful!")

import os
filename  = os.path.basename(__file__)
print(filename)