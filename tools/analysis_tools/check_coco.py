#%%
import copy
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import json
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

json_file = '/ailab_mat/dataset/FLIR_ADAS_v2/video_rgb_test/coco.json'
# %%
with open(json_file, 'r') as f:
    coco_data = json.load(f)

#%%
print(coco_data.keys())
#%%
edited_json = '/ailab_mat/dataset/FLIR_ADAS_v2/video_rgb_test/test_coco_v3.json'
# %%
with open(edited_json, 'r') as f:
    coco_data_edited = json.load(f)

# %%
print(coco_data_edited.keys())
# %%
print(coco_data_edited.keys() == coco_data.keys())
# %%
