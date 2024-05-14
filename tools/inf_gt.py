import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from tqdm import tqdm
# Assuming you have pycocotools installed for dealing with COCO data
from pycocotools.coco import COCO

# Initialize COCO api for instance annotations
coco = COCO('/ailab_mat/dataset/FLIR_ADAS_v2/images_rgb_train/coco.json')
root = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/inferences/gt_inference/flir-rgb'
#Get total number of images in the dataset
img_ids = coco.getImgIds()
print('Total number of images:', len(img_ids))



for i in tqdm(range(len(img_ids))):
    # Specify the image to display
    img_id = i  # Replace with your image id
    img_info = coco.loadImgs(img_id)[0]
    img_path = '/ailab_mat/dataset/FLIR_ADAS_v2/images_rgb_train/' + img_info['file_name']

    # Load and display image
    image = Image.open(img_path)
    # Load and display instance annotations
    ax = plt.gca()
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    annotations = coco.loadAnns(ann_ids)
    for ann in annotations:
        # Draw rectangle for the bounding box
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Display class name
        cat = coco.loadCats(ann['category_id'])[0]['name']
        plt.text(x, y, cat, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
        plt.imshow(image)
        plt.savefig(root + '/gt_' + img_info['file_name'].replace('data/', '').replace('.jpg', '.png'))
