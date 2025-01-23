# Description: Check the FLIR dataset with visualize

from pycocotools.coco import COCO
import cv2
import os
import numpy as np
from check_flir import get_category_names, draw_bbox

COLOR_PALETTE = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255]])

def visualize(coco_path, root, flag):
    if flag == 'val':
        flag = 'test'
    rgb_root = os.path.join(root, 'visible')
    thermal_root = os.path.join(root,'infrared')
    coco = COCO(coco_path)
    
    # Get category names from annotation file
    category_names = get_category_names(coco)
    print("Categories in dataset:", category_names)
    
    img_ids = coco.getImgIds()
    current_idx = 0
    
    while True:
        # Get current image info
        img_info = coco.loadImgs(img_ids[current_idx])[0]
        img_name = img_info['file_name']
        thermal_name = img_info['file_name']
        
        # Load RGB and thermal images
        rgb_path = os.path.join(rgb_root,flag,  img_name)
        thermal_path = os.path.join(thermal_root,flag, thermal_name)
        
        rgb_img = cv2.imread(rgb_path)
        thermal_img = cv2.imread(thermal_path)
        
        if rgb_img is None or thermal_img is None:
            print(f"Error loading images: {img_name}")
            current_idx = (current_idx + 1) % len(img_ids)
            continue
            
        # Get annotations for current image
        ann_ids = coco.getAnnIds(imgIds=img_info['id'])
        anns = coco.loadAnns(ann_ids)
        
        # Draw bounding boxes
        rgb_vis = rgb_img.copy()
        thermal_vis = thermal_img.copy()
        
        for ann in anns:
            bbox = ann['bbox']
            category_id = ann['category_id']
            color = COLOR_PALETTE[category_id % len(COLOR_PALETTE)]
            rgb_vis = draw_bbox(rgb_vis, bbox, category_id, color, category_names)
            thermal_vis = draw_bbox(thermal_vis, bbox, category_id, color, category_names)
        
        # Display images side by side
        combined_img = np.hstack((rgb_vis, thermal_vis))
        cv2.imshow('FLIR Dataset Visualization (Press q to quit, <- or -> to navigate)', combined_img)
        
        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == 83 or key == ord('d'):  # Right arrow or 'd'
            current_idx = (current_idx + 1) % len(img_ids)
        elif key == 81 or key == ord('a'):  # Left arrow or 'a'
            current_idx = (current_idx - 1) % len(img_ids)
            
    cv2.destroyAllWindows()


def main():
    coco_path = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/LLVIP/coco_annotations/train.json'
    root = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/LLVIP/'
    flag = os.path.basename(coco_path).split('.')[0]
    visualize(coco_path, root, flag)


if __name__ == "__main__":
    main()
