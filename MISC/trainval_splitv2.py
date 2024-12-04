import os
import json
import cv2
from pathlib import Path
from datetime import datetime
import random
import shutil
from pycocotools.coco import COCO
import numpy as np
import natsort

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
            # shutil.copy(src_img, dst_img)
            image_id += 1

    # Use the consistent category mapping for merged categories
    categories = [{'id': cat_id, 'name': cat_name} for cat_name, cat_id in mapped_category.items()]
    ann_dict = {
        'images': all_images,
        'annotations': all_annotations,
        'categories': categories
    }
    return ann_dict

def frame_split_process(ann, train_ratio):
    '''
    Split coco annotation by ratio
    '''


    total_images = len(ann['images'])
    total_annotations = len(ann['annotations'])
    train_num = int(total_images * train_ratio)
    val_num = len(ann['images']) - train_num
    
    id_idx = np.arange(1, total_images+1)
    space = 2
    val_id_idx = np.arange(2, space * val_num, space)
    train_id_idx = np.array([i for i in id_idx if i not in val_id_idx])

    train_images = [img for img in ann['images'] if img['id'] in train_id_idx]
    val_images = [img for img in ann['images'] if img['id'] in val_id_idx]
    train_anns = [ann for ann in ann['annotations'] if ann['image_id'] in train_id_idx]
    val_anns = [ann for ann in ann['annotations'] if ann['image_id'] in val_id_idx]

    train_ann = {
        'images': train_images,
        'annotations': train_anns,
        'categories': ann['categories']
    }

    val_ann = {
        'images': val_images,
        'annotations': val_anns,
        'categories': ann['categories']
    }
    return train_ann, val_ann


def random_split_process(folders, train_ratio):
    '''
    Split coco annotation by ratio
    '''
    train_image_id = 1
    train_ann_id = 1
    val_image_id = 1
    val_ann_id = 1


    all_train_imgs = []
    all_val_imgs = []
    all_train_anns = []
    all_val_anns = []
    for folder in folders:
        # Load annotations
        ann_file = folder / "annotations.json"
        with open(ann_file, 'r') as f:
            data = json.load(f)
        coco = COCO(ann_file)

        total_images = len(data['images'])
        total_annotations = len(data['annotations'])
        train_num = int(total_images * train_ratio)
        val_num = len(data['images']) - train_num
    
        id_idx = np.arange(0, total_images)
        train_id_idx = natsort.natsorted(np.random.choice(id_idx, train_num, replace=False))
        val_id_idx = natsort.natsorted(np.array([i for i in id_idx if i not in train_id_idx]))

        for train_id in train_id_idx:
            # img = [img for img in data['images'] if img['id'] == train_id][0]
            img = coco.loadImgs(int(train_id))[0]
            img['id'] = train_image_id
            img['file_name'] = f"rgb_images/{folder.name}_{img['file_name'].split('/')[-1]}"
            all_train_imgs.append(img)
            # anns = [ann for ann in data['annotations'] if ann['image_id'] == train_id]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=train_id))
            for ann in anns:
                ann['id'] = train_ann_id
                ann['image_id'] = train_image_id
                all_train_anns.append(ann)
                train_ann_id += 1
            train_image_id += 1
        
        for val_id in val_id_idx:
            # img = [img for img in ann['images'] if img['id'] == val_id][0]
            img = coco.loadImgs(int(val_id))[0]
            img['id'] = val_image_id
            img['file_name'] = f"rgb_images/{folder.name}_{img['file_name'].split('/')[-1]}"
            all_val_imgs.append(img)
            anns = coco.loadAnns(coco.getAnnIds(imgIds=val_id))
            for ann in anns:
                ann['id'] = val_ann_id
                ann['image_id'] = val_image_id
                all_val_anns.append(ann)
                val_ann_id += 1
            val_image_id += 1
    
    train_ann = {
        'images': all_train_imgs,
        'annotations': all_train_anns,
        'categories': data['categories']
    }

    val_ann = {
        'images': all_val_imgs,
        'annotations': all_val_anns,
        'categories': data['categories']
    }
    return train_ann, val_ann

def frame_last_ratio_process(folders, train_ratio):
    '''
    Split coco annotation by ratio with last frame
    '''
    train_image_id = 1
    train_ann_id = 1
    val_image_id = 1
    val_ann_id = 1


    all_train_imgs = []
    all_val_imgs = []
    all_train_anns = []
    all_val_anns = []
    for folder in folders:
        # Load annotations
        ann_file = folder / "annotations.json"
        with open(ann_file, 'r') as f:
            data = json.load(f)
        coco = COCO(ann_file)

        total_images = len(data['images'])
        total_annotations = len(data['annotations'])
        train_num = int(total_images * train_ratio)
        val_num = len(data['images']) - train_num
    
        id_idx = np.arange(0, total_images)
        # train_id_idx = natsort.natsorted(np.random.choice(id_idx, train_num, replace=False))
        # val_id_idx = natsort.natsorted(np.array([i for i in id_idx if i not in train_id_idx]))
        train_id_idx = id_idx[:train_num]
        val_id_idx = id_idx[train_num:]

        for train_id in train_id_idx:
            # img = [img for img in data['images'] if img['id'] == train_id][0]
            img = coco.loadImgs(int(train_id))[0]
            img['id'] = train_image_id
            img['file_name'] = f"rgb_images/{folder.name}_{img['file_name'].split('/')[-1]}"
            all_train_imgs.append(img)
            # anns = [ann for ann in data['annotations'] if ann['image_id'] == train_id]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=train_id))
            for ann in anns:
                ann['id'] = train_ann_id
                ann['image_id'] = train_image_id
                all_train_anns.append(ann)
                train_ann_id += 1
            train_image_id += 1
        
        for val_id in val_id_idx:
            # img = [img for img in ann['images'] if img['id'] == val_id][0]
            img = coco.loadImgs(int(val_id))[0]
            img['id'] = val_image_id
            img['file_name'] = f"rgb_images/{folder.name}_{img['file_name'].split('/')[-1]}"
            all_val_imgs.append(img)
            anns = coco.loadAnns(coco.getAnnIds(imgIds=val_id))
            for ann in anns:
                ann['id'] = val_ann_id
                ann['image_id'] = val_image_id
                all_val_anns.append(ann)
                val_ann_id += 1
            val_image_id += 1
    
    train_ann = {
        'images': all_train_imgs,
        'annotations': all_train_anns,
        'categories': data['categories']
    }

    val_ann = {
        'images': all_val_imgs,
        'annotations': all_val_anns,
        'categories': data['categories']
    }
    return train_ann, val_ann


def split_by_random(root: Path, train_ratio=0.8, file_prefix = "frameRandom_"):
    '''
    Random split with total frames
    '''
    
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith('drone')]
    train_ann, val_ann = random_split_process(folders, train_ratio)

    with open(output_dir / f"{file_prefix}train.json", 'w') as f:
        json.dump(train_ann, f, indent=2)
    with open(output_dir / f"{file_prefix}val.json", 'w') as f:
        json.dump(val_ann, f, indent=2)


def split_by_frame(root: Path, train_ratio=0.8, file_prefix = "frame_"):
    '''
    Train test split with frame level
    '''
    
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith('drone')]
    ann = split_process(folders, root)

    train_ann, val_ann = frame_split_process(ann, train_ratio)
    with open(output_dir / f"{file_prefix}train.json", 'w') as f:
        json.dump(train_ann, f, indent=2)
    with open(output_dir / f"{file_prefix}val.json", 'w') as f:
        json.dump(val_ann, f, indent=2)


def split_by_lastframe(root: Path, train_ratio=0.8, file_prefix = "frameVideo_"):
    '''
    Train test split with frame sequentially
    '''
    
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith('drone')]
    train_ann, val_ann = frame_last_ratio_process(folders, train_ratio)

    with open(output_dir / f"{file_prefix}train.json", 'w') as f:
        json.dump(train_ann, f, indent=2)
    with open(output_dir / f"{file_prefix}val.json", 'w') as f:
        json.dump(val_ann, f, indent=2)

def split_by_file(root: Path, train_ratio=0.8, file_prefix = 'file_'):
    """
    Merge annotations from multiple folders and split into train/val sets
    
    Args:
        root: Root directory containing output folders
        train_ratio: Ratio of images to use for training (default 0.8)
    """
    # Get all output folders
    output_dir = root / "output"
    folders = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith('drone')]
    val_folders = [folders[1], folders[6]]
    train_folders = [f for f in folders if f not in val_folders]

    train_ann = split_process(train_folders, root)
    val_ann = split_process(val_folders, root)

    with open(output_dir / "video_train.json", 'w') as f:
        json.dump(train_ann, f, indent=2)
    
    with open(output_dir / "video_val.json", 'w') as f:
        json.dump(val_ann, f, indent=2)





def main():
    root = Path("/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/241108-indoor-gist/")
    split_by_file(root)


if __name__ == "__main__":
    main()
