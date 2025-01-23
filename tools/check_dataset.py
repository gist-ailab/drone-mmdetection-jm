#%%
import numpy as np
import cv2
import glob
import json
from pycocotools.coco import COCO
import os

pth = '/SSDb/jemo_maeng/dset/data/FLIR_aligned_coco/annotations/val.json'

coco = COCO(pth)
img_ids = coco.getImgIds()
ann_ids = coco.getAnnIds(img_ids[0])

img_id = img_ids[0]
ann_ids = coco.getAnnIds(img_id)
img_info = coco.loadImgs(img_id)[0]
ann_info = coco.loadAnns(ann_ids)



root = '/SSDb/jemo_maeng/dset/data/FLIR_aligned_coco/'
prefix = 'val_RGB'
thermal_prefix = 'val_thermal'  


rgb_pth = os.path.join(root, prefix, img_info['file_name_IR'])
rgb_img = cv2.imread(rgb_pth)

thermal_pth = os.path.join(root, thermal_prefix, img_info['file_name_IR'])
thermal_img = cv2.imread(thermal_pth, cv2.IMREAD_GRAYSCALE)
print('debug')