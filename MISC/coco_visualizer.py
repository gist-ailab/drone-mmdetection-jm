# coco visualizer

root = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/241108-indoor-gist/output'
json_pth = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/241108-indoor-gist/output/train.json'
output_dir = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/241108-indoor-gist/output/viz_anno'

import os
import json
import cv2
from pathlib import Path
from pycocotools.coco import COCO

# Load the COCO dataset
coco = COCO(json_pth)

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define a color mapping for different classes
color_mapping = {
    1: (255, 0, 0),  # Red for class 1
    2: (0, 255, 0),  # Green for class 2
    3: (0, 0, 255),  # Blue for class 3
    4: (255, 255, 0),  # Yellow for class 4
    5: (0, 255, 255),  # Cyan for class 5
}

# Visualize the images with bounding boxes and class
for img_id in coco.getImgIds():
    img = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(ann_ids)
    image = cv2.imread(os.path.join(root, img['file_name']))
    for ann in anns:
        bbox = ann['bbox']
        class_id = ann['category_id']
        print(class_id)
        class_name = coco.loadCats(class_id)[0]['name']
        # Use the color mapping for the bounding box
        color = color_mapping.get(class_id, (255, 0, 0))  # Default to red if class_id not found
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), color, 2)
        cv2.putText(image, class_name, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, img['file_name'].split('/')[-1]), image)

