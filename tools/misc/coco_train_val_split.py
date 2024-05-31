'''
Train test split for coco dataset

args:
    ann_file: str, path to annotation file
    save_path: str, path to save the split
    train_ratio: float, ratio of training set
'''
# import os
# import json
# import random

def train_test_split_coco(ann_file, train_ratio, prefix):
    save_path = os.path.dirname(ann_file)
    # Ensure the train_ratio is between 0 and 1
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")

    # Load the COCO annotation file
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # Extract images and annotations
    images = coco_data['images']
    annotations = coco_data['annotations']
    info = coco_data['info']
    categories = coco_data['categories']
    licenses = coco_data['licenses']

    # Shuffle images to ensure random splitting
    # random.shuffle(images)
    # Calculate split index
    split_index = int(train_ratio * len(images))

    # Split images into training and test sets
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create a lookup for image ids in train and test sets
    train_image_ids = {img['id'] for img in train_images}
    test_image_ids = {img['id'] for img in test_images}

    # Split annotations into training and test sets based on image ids
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    test_annotations = [ann for ann in annotations if ann['image_id'] in test_image_ids]

    # Prepare training and test data
    train_data = {
        'info': info,
        'licenses': licenses,
        'categories': categories,
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories']
    }

    test_data = {
        'info': info,
        'licenses': licenses,
        'categories': categories,
        'images': test_images,
        'annotations': test_annotations,
        'categories': coco_data['categories']
    }

    # Save the split datasets
    train_file = os.path.join(save_path, 'train_{}.json'.format(prefix))
    test_file = os.path.join(save_path, 'test_{}.json'.format(prefix))

    with open(train_file, 'w') as f:
        json.dump(train_data, f)

    with open(test_file, 'w') as f:
        json.dump(test_data, f)

    print(f"[INFO]: Training and test sets saved to {save_path}")



#%%
import numpy as np
import os, json, cv2, random
from pycocotools.coco import COCO
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as f:
        json.dump({"info": info, "licenses": licenses, "images": images, "annotations": annotations, "categories": categories}, f, ensure_ascii=False, indent=4, sort_keys = True)
    print(f"Saved to {file}")

def get_ann(idx, list):
    anns = coco.getAnnIds(idx)
    for j in range(len(anns)):
        ann = coco.loadAnns(anns[j])[0]
        list.append(ann)
    return list


#%%

save_path = '/ailab_mat/dataset/FLIR_ADAS_v2/video_rgb_test'
path = os.path.join(save_path,'coco.json')
with open(path, 'r') as f:
    inf_json = json.load(f)
# %%
info = inf_json['info']
categories = inf_json['categories']
images = inf_json['images']
annotations = inf_json['annotations']
licenses = inf_json['licenses']
len_images = len(images)

for i in range(len(categories)):
    n = categories[i]["name"]
    if len(n.split('.'))==2:
        categories[i].clear()
categories = list(filter(None, categories))  #remove duplicated categories
#%%

# %%
train_categories, val_categories  = categories, categories
img_num = len(images)
ratio = 0.8
split_index = int(ratio * len(images))
train_images = images[:split_index]
val_images = images[split_index:]
#%%
coco = COCO(path)
#%%
train_anns = []
val_anns = []
# %%
for i in tqdm(range(len(train_images))):
    train_anns = get_ann(i, train_anns)

for i in tqdm(range(len(val_images))):
    idx = i+split_index
    val_anns = get_ann(idx, val_anns)
# %%
save_coco(save_path+'/train_coco_v3.json', info, licenses, train_images, train_anns, categories)
save_coco(save_path+'/test_coco_v3.json', info, licenses, val_images, val_anns, categories)