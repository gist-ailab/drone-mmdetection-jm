import json
from pycocotools.coco import COCO
import random

# Load the original annotation file
ann_file = "/SSDb/jemo_maeng/dset/data/DroneDataV2/FLIR-align/annotations/train.json"
output_file = "/SSDb/jemo_maeng/dset/data/DroneDataV2/FLIR-align/annotations/train_small.json"

# Load COCO format annotations
with open(ann_file, 'r') as f:
    data = json.load(f)

# Initialize COCO api
coco = COCO(ann_file)

# Get all image ids
img_ids = coco.getImgIds()

# Randomly select 100 images
selected_img_ids = random.sample(img_ids, min(100, len(img_ids)))

# Create new data structure
small_dataset = {
    'categories': data['categories'],
    'images': [],
    'annotations': []
}

# Get selected images
selected_images = []
selected_ann_ids = []

for img_id in selected_img_ids:
    # Add image info
    img_info = coco.loadImgs(img_id)[0]
    small_dataset['images'].append(img_info)
    
    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    selected_ann_ids.extend(ann_ids)

# Add selected annotations
for ann_id in selected_ann_ids:
    ann = coco.loadAnns(ann_id)[0]
    small_dataset['annotations'].append(ann)

# Save the small dataset
with open(output_file, 'w') as f:
    json.dump(small_dataset, f)

print(f"Created small dataset with {len(small_dataset['images'])} images and {len(small_dataset['annotations'])} annotations")
print(f"Saved to: {output_file}")
