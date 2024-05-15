import mmcv
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mmdet.apis import init_detector, inference_detector
from torch.utils.data import DataLoader
from mmdet.datasets import CocoDataset  # or the specific dataset class you need
# from mmdet.datasets import flair-adas_detection


from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# cfg = mmcv.Config.fromfile('/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/configs/custom/faster-rcnn_r50_fpn_2x_flair-adas.py')
config_file = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/configs/custom/faster-rcnn_r50_fpn_2x_flair-adas.py'
cfg = Config.fromfile(config_file)

ckpt_pth = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/faster-rcnn_r50_fpn_2x_flair-adas/epoch_24.pth'
flag = 'flair-adas-rgb'
save_pth = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/inferences/{}'.format(flag)
os.makedirs(save_pth, exist_ok=True)

# model = init_detector(cfg, ckpt_pth, device='cuda:0')
model = init_detector(config_file, ckpt_pth, device='cuda:0')

# Build the dataset and data_loader
dataset = CocoDataset(cfg.data.val)
data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=cfg.data.workers_per_gpu,
    collate_fn=lambda batch: {key: [d[key] for d in batch] for key in batch[0]}
)

# outputs = single_gpu_test(model, data_loader, show=False)
outputs = []

for i, data in enumerate(data_loader):
    with torch.no_grad():
        result = inference_detector(model, data['img'][0])
    outputs.append(result)


eval_results = dataset.evaluate(outputs, metric='mAP', logger=None)
mean_ap = eval_results['bbox_mAP']  # Change 'bbox_mAP' if using different evaluation metrics
# Evaluate mAP and print it
eval_results = dataset.evaluate(outputs, metric='mAP', logger=None)
mean_ap = eval_results['bbox_mAP']  # Change 'bbox_mAP' if using different evaluation metrics
print(f"mAP: {mean_ap}")

# Save mAP to a file
with open(f"{save_pth}/{flag}_map_results.txt", "w") as f:
    f.write(f"mAP: {mean_ap}\n")

# Example to generate a confusion matrix
gt_labels = []
pred_labels = []

# Loop through the data in the data loader
for i, data in enumerate(data_loader):
    gt_labels.extend([int(label) for label in data['gt_labels'].data[0].cpu().numpy()])
    pred_labels.extend([int(np.argmax(output['scores'])) for output in outputs[i]])

# Calculate confusion matrix
conf_matrix = confusion_matrix(gt_labels, pred_labels)
print(conf_matrix)

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_title('Confusion Matrix', pad=20)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')

# Adding text annotations
threshold = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(j, i, format(conf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > threshold else "black")

# Save confusion matrix to a file
# plt.savefig("confusion_matrix.png")
plt.savefig(f"{save_pth}/{flag}_confusion_matrix.png")
plt.close()