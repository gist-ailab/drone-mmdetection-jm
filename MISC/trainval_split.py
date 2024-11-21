import os
import json
import cv2
from pathlib import Path
from datetime import datetime
import random
import shutil

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
            
        # Update image paths and IDs
        for img in data['images']:
            img['id'] = image_id
            old_file_name = img['file_name']
            file_name = old_file_name.split('/')[-1]
            # img['file_name'] = f"{folder.name}_{old_file_name}"
            img['file_name'] = f"rgb_images/{folder.name}_{file_name}"
            all_images.append(img)
            # Copy image to new location
            src_img = folder  / old_file_name
            dst_img = save_pth / "rgb_images" / f"{folder.name}_{file_name}"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_img, dst_img)
            image_id += 1
            
        # Update annotation image IDs
        for ann in data['annotations']:
            ann['id'] = ann_id
            ann['image_id'] = image_id - 1  # Link to the last added image
            all_annotations.append(ann)
            ann_id += 1

            # Create merged annotation structure
    merged_ann = {
        'images': all_images,
        'annotations': all_annotations,
        'categories': data['categories']  # Use categories from last loaded file
    }

    num_images = len(all_images)


    ann_dict ={
        'images': all_images,
        'annotations': all_annotations,
        'categories': data['categories']
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

    '''
    # # Split images
    # for idx, img in enumerate(merged_ann['images']):
    #     if idx in train_indices:
    #         train_ann['images'].append(img)
    #     else:
    #         val_ann['images'].append(img)
    
    # # Split annotations based on image_id
    # train_image_ids = {img['id'] for img in train_ann['images']}
    # for ann in merged_ann['annotations']:
    #     if ann['image_id'] in train_image_ids:
    #         train_ann['annotations'].append(ann)
    #     else:
    #         val_ann['annotations'].append(ann)
    '''

    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_ann, f, indent=2)
    
    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_ann, f, indent=2)


def main():
    root = Path("/media/ailab/HDD1/Workspace/Download/Drone-detection/241108-indoor-gist")
    trainval_split(root)


if __name__ == "__main__":
    main()