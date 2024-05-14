from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mmcv
from tqdm import tqdm
import os
classes = ('person', 'bike', 'car', 'motor', 'airplane', 'bus', 'train', 'truck', 'boat', 'light', 'hydrant', 'sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'deer', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'stroller', 'rider', 'scooter', 'vase', 'scissors', 'face', 'other vehicle', 'license plate')

def show_predictions_and_gt(img, result, gt_annotations, category_names, save_root):
    filename = os.path.basename(img)
    img = mmcv.imread(img)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    ax = plt.gca()

    # Extracting data from the DetDataSample object
    if hasattr(result, 'pred_instances'):
        # Assuming pred_instances has the properties: bboxes, scores, labels
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()

        # Draw predictions
        for bbox, score, label in zip(bboxes, scores, labels):
            if score > 0.3:
                x1, y1, x2, y2 = bbox
                width, height = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f'Pred: {category_names[label]} {score:.2f}', color='green', fontsize=8)
    else:
        print("No predictions found")

    # Draw ground truth
    for ann in gt_annotations:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f"GT: {category_names[ann['category_id']]}", color='blue', fontsize=8)

    #bgr 2 rgb
    img = img[..., ::-1]
    plt.axis('off')
    plt.show()
    plt.savefig(os.path.join(save_root, filename))
    plt.close()
    

save_root = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/inferences/gt_inference/flir-rgb-gt'
os.makedirs(save_root, exist_ok=True)

# Path to the configuration file and model weights
config_file = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/configs/custom/faster-rcnn_r50_fpn_2x_flair-adas.py'
checkpoint_file= '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/faster-rcnn_r50_fpn_2x_flair-adas/epoch_24.pth'
# Initialize the detector
model = init_detector(config_file, checkpoint_file, device='cuda:0')
coco = COCO('/ailab_mat/dataset/FLIR_ADAS_v2/images_rgb_train/coco.json')

for img_id in tqdm(coco.getImgIds()):
    img_info = coco.loadImgs(img_id)[0]
    img_path = f"/ailab_mat/dataset/FLIR_ADAS_v2/images_rgb_train/{img_info['file_name']}"
    

    img_ = mmcv.imread(img_path)
    # Run inference
    result = inference_detector(model, img_path)

    # Load ground truth annotations
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    gt_annotations = coco.loadAnns(ann_ids)

    # Visualize
    show_predictions_and_gt(img_path, result, gt_annotations, classes, save_root=save_root)  # Define save_root based on your setup