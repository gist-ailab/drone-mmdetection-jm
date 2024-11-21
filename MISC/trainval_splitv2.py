import os
import json
import cv2
from pathlib import Path
from datetime import datetime
import random
import shutil
from pycocotools.coco import COCO

# Define a consistent category mapping
mapped_category = {
    'human': 1,
    'Door': 2,
    'fire extinguisher': 3,
    'exit pannel': 4,
    'window': 5,
    # Add other categories as needed
}


def split_process(folders, root):
    all_images = []
    all_annotations = []
    image_id = 1
    ann_id = 1

    save_pth = root / "output"

    for folder in folders:
        # Load annotations
        ann_file = folder / "annotations.json"
        with open(ann_file, 'r') as f:
            data = json.load(f)
        coco = COCO(ann_file)
        # Load image and annotation data using pycocotools
        cur_categories = coco.loadCats(coco.getCatIds())
        category_dict = {cat['id']: cat['name'] for cat in cur_categories}

        img_ids = coco.getImgIds()
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            old_file_name = img_info['file_name']
            file_name = img_info['file_name'].split('/')[-1]
            img_processed ={
                'id': image_id,
                'file_name' : f"rgb_images/{folder.name}_{file_name}",
                'width': img_info['width'],
                'height': img_info['height'],
                'date_captured': datetime.now().isoformat()
            }
            all_images.append(img_processed)

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                ann['id'] = ann_id
                ann['image_id'] = image_id
                old_name = category_dict[ann['category_id']]
                new_category_id = mapped_category[old_name]
                ann['category_id'] = new_category_id
                all_annotations.append(ann)
                ann_id += 1
            src_img = folder / old_file_name
            dst_img = save_pth / "rgb_images" / f"{folder.name}_{file_name}"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_img, dst_img)
            image_id += 1

    # Use the consistent category mapping for merged categories
    categories = [{'id': cat_id, 'name': cat_name} for cat_name, cat_id in mapped_category.items()]
    # categories = data['categories']
    ann_dict = {
        'images': all_images,
        'annotations': all_annotations,
        'categories': categories
    }
    return ann_dict

def trainval_split(root: Path, train_ratio=0.8):
    """
    Merge annotations from multiple folders and split into train/val sets
    
    Args:
        root: Root directory containing output folders
        train_ratio: Ratio of images to use for training (default 0.8)
    """
    # Get all output folders
    output_dir = root / "output"
    folders = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith('drone')]
    train_folders = random.sample(folders, int(len(folders) * train_ratio))
    val_folders = [f for f in folders if f not in train_folders]
    train_ann = split_process(train_folders, root)
    val_ann = split_process(val_folders, root)

    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_ann, f, indent=2)
    
    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_ann, f, indent=2)


def main():
    root = Path("/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/241108-indoor-gist")
    trainval_split(root)


if __name__ == "__main__":
    main()
