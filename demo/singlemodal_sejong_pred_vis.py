#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import colorsys
import mmengine
import json
from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from typing import Tuple, Dict, List, Optional
import wandb

# 사용자의 커스텀 모듈 등록 (필요 시)
# from mcdet import *

class RGBInferenceVisualizer:
    """
    RGB 이미지 전용 CocoDataset에 대한 추론, 시각화, 예측 저장을 수행하는 클래스.
    """
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.cfg = mmengine.Config.fromfile(config_path)
        mmengine.registry.init_default_scope(self.cfg.get('default_scope', 'mmdet'))
        self.model = init_detector(config_path, checkpoint_path, device=device, cfg_options={'strict': False})
        
        self.classes = self.cfg.val_dataloader.dataset.metainfo['classes']
        self.colors = self.cfg.val_dataloader.dataset.metainfo.get('palette')
        # 설정 파일에 팔레트가 없는 경우를 대비해 랜덤 색상 생성
        if self.colors is None:
            print("Warning: Palette not found in config. Generating random colors.")
            np.random.seed(42)
            self.colors = np.random.randint(0, 256, size=(len(self.classes), 3)).tolist()

    def crop_and_adjust_bboxes(self, boxes: np.ndarray, labels: np.ndarray, crop_box: List[int],
                               scores: Optional[np.ndarray] = None) -> Tuple:
        """시각화를 위해 Bbox를 자르고 좌표를 Crop된 이미지 기준으로 변환합니다."""
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        clipped_boxes = boxes.copy()
        clipped_boxes[:, 0] = np.maximum(clipped_boxes[:, 0], crop_x1)
        clipped_boxes[:, 1] = np.maximum(clipped_boxes[:, 1], crop_y1)
        clipped_boxes[:, 2] = np.minimum(clipped_boxes[:, 2], crop_x2)
        clipped_boxes[:, 3] = np.minimum(clipped_boxes[:, 3], crop_y2)

        valid_widths = clipped_boxes[:, 2] - clipped_boxes[:, 0]
        valid_heights = clipped_boxes[:, 3] - clipped_boxes[:, 1]
        valid_indices = (valid_widths > 0) & (valid_heights > 0)
        
        if not np.any(valid_indices):
            empty_scores = np.empty((0,)) if scores is not None else None
            result = (np.empty((0, 4)), np.empty((0,), dtype=np.int64))
            return result + (empty_scores,) if scores is not None else result

        final_boxes = clipped_boxes[valid_indices]
        final_labels = labels[valid_indices]
        final_boxes[:, [0, 2]] -= crop_x1
        final_boxes[:, [1, 3]] -= crop_y1

        if scores is not None:
            return final_boxes, final_labels, scores[valid_indices]
        else:
            return final_boxes, final_labels

    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                   text_prefix: str, box_format: str = 'xyxy') -> np.ndarray:
        """단일 이미지에 Bbox를 그립니다."""
        output_img = image.copy()
        for bbox, label in zip(boxes, labels):
            if box_format == 'xywh':
                x, y, w, h = bbox.astype(int); x1, y1, x2, y2 = x, y, x + w, y + h
            else: # xyxy
                x1, y1, x2, y2 = bbox.astype(int)

            class_name = self.classes[label]; color = self.colors[label % len(self.colors)]
            label_text = f'{text_prefix}: {class_name}'
            
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(output_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return output_img
    
    def draw_translucent_boxes(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                               text_prefix: str, box_format: str = 'xyxy', alpha: float = 0.5) -> np.ndarray:
        """단일 이미지에 반투명 Bbox를 그립니다."""
        output_img = image.copy(); overlay = output_img.copy()
        draw_info = []
        for bbox, label in zip(boxes, labels):
            if box_format == 'xywh': x, y, w, h = bbox.astype(int); x1, y1, x2, y2 = x, y, x + w, y + h
            else: x1, y1, x2, y2 = bbox.astype(int)
            r, g, b = self.colors[label % len(self.colors)]; hls = colorsys.rgb_to_hls(r / 255., g / 255., b / 255.)
            lightness = min(1.0, hls[1] * 1.3); new_rgb = colorsys.hls_to_rgb(hls[0], lightness, hls[2])
            gt_color = (int(new_rgb[0] * 255), int(new_rgb[1] * 255), int(new_rgb[2] * 255))
            label_text = f'{text_prefix}: {self.classes[label]}'
            draw_info.append({'coords': (x1, y1, x2, y2), 'color': gt_color, 'text': label_text})
        for info in draw_info:
            x1, y1, x2, y2 = info['coords']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), info['color'], -1)
        cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0, output_img)
        for info in draw_info:
            x1, y1, x2, y2 = info['coords']; gt_color = info['color']; label_text = info['text']
            cv2.rectangle(output_img, (x1, y1), (x2, y2), gt_color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_img, (x1, y1 - text_h - 5), (x1 + text_w, y1), gt_color, -1)
            cv2.putText(output_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return output_img

    @torch.no_grad()
    def run_inference_and_visualize(self, output_dir: str, num_samples: int, score_threshold: float,
                                      output_json_path: str, use_vis_crop: bool):
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'w_gt'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'wo_gt'), exist_ok=True)
        
        val_dataset_cfg = self.cfg.val_dataloader.dataset
        dataset = DATASETS.build(val_dataset_cfg)
        dataset.full_init()
        
        total_samples = len(dataset) if num_samples < 0 else min(num_samples, len(dataset))

        print(f"'{val_dataset_cfg.type}' dataset contains {len(dataset)} samples.")
        print(f"Processing {total_samples} samples for visualization and prediction saving...")

        crop_box = [40, 110, 480, 480] if use_vis_crop else None
        if crop_box:
            print(f"All visualization bboxes will be cropped to: {crop_box}")

        self.model.eval()
        coco_results = []

        for i, data in enumerate(dataset):
            if i >= total_samples: break
            
            original_data_sample = data['data_samples']
            image_id = original_data_sample.img_id
            
            batched_data = { 'inputs': [data['inputs']], 'data_samples': [original_data_sample] }
            processed_data = self.model.data_preprocessor(batched_data, training=False)
            predictions = self.model.forward(**processed_data, mode='predict')
            pred_sample = predictions[0]
            
            scale_factor = processed_data['data_samples'][0].metainfo['scale_factor']
            pred_instances = pred_sample.pred_instances[pred_sample.pred_instances.scores > score_threshold]
            
            # --- 1. Get predictions in original image coordinate space (xyxy) ---
            pred_boxes_xyxy = pred_instances.bboxes.cpu().numpy() / np.tile(scale_factor, 2)
            pred_labels = pred_instances.labels.cpu().numpy()
            pred_scores = pred_instances.scores.cpu().numpy()

            # --- 2. Save predictions to COCO format list ---
            for box, label, score in zip(pred_boxes_xyxy, pred_labels, pred_scores):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                category_id = dataset.cat_ids[label]
                coco_results.append({
                    'image_id': image_id, 'category_id': category_id,
                    'bbox': [float(coord) for coord in [x1, y1, w, h]],
                    'score': float(score)
                })
            
            # --- 3. Prepare for visualization ---
            img_path = original_data_sample.img_path
            img_id_str = Path(img_path).stem
            print(f"[{i+1}/{total_samples}] Processing: {img_id_str}")

            vis_image = cv2.imread(img_path)

            # Get original GT boxes (already in xyxy format)
            gt_instances = original_data_sample.gt_instances
            if 'bboxes' in gt_instances:
                gt_boxes_xyxy = gt_instances.bboxes.cpu().numpy()
                gt_labels = gt_instances.labels.cpu().numpy()
            else:
                gt_boxes_xyxy, gt_labels = np.empty((0, 4)), np.empty((0,))

            if use_vis_crop:
                # If cropping, transform both GT and Pred boxes for visualization
                vis_gt_boxes, vis_gt_labels = self.crop_and_adjust_bboxes(gt_boxes_xyxy, gt_labels, crop_box)
                vis_pred_boxes, vis_pred_labels, _ = self.crop_and_adjust_bboxes(
                    pred_boxes_xyxy, pred_labels, crop_box, scores=pred_scores)
                
                # Crop the image itself for visualization
                cr_x1, cr_y1, cr_x2, cr_y2 = crop_box
                vis_image = vis_image[cr_y1:cr_y2, cr_x1:cr_x2]
            else:
                # If not cropping, use original boxes for visualization
                vis_gt_boxes, vis_gt_labels = gt_boxes_xyxy, gt_labels
                vis_pred_boxes, vis_pred_labels = pred_boxes_xyxy, pred_labels

            # Draw predictions
            result_img = self.draw_boxes(vis_image.copy(), vis_pred_boxes, vis_pred_labels, "Pred", 'xyxy')
            cv2.imwrite(os.path.join(output_dir, 'wo_gt', f'{img_id_str}.jpg'), result_img)
            
            # Draw ground truth
            result_img_w_gt = self.draw_translucent_boxes(result_img, vis_gt_boxes, vis_gt_labels, "GT", 'xyxy', alpha=0.5)
            cv2.imwrite(os.path.join(output_dir, 'w_gt', f'{img_id_str}.jpg'), result_img_w_gt)

        print(f"\nSaved {len(coco_results)} predictions to '{output_json_path}'.")
        with open(output_json_path, 'w') as f:
            json.dump(coco_results, f, indent=4)
        
        print("\nVisualization and prediction saving complete.")

def main():
    parser = argparse.ArgumentParser(description='RGB CocoDataset Visualization and Prediction Saving')
    parser.add_argument(
        '--config', 
        default='/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_faster_rcnn__v2/hinton-sejong2504_faster_rcnn_lr0.01_ep50_v2.py',
        help='Model config file path'
    )
    parser.add_argument(
        '--checkpoint',
        default='/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_faster_rcnn__v2/epoch_20.pth',
        help='Model checkpoint file path'
    )
    parser.add_argument('--output-dir', default='/ailab_mat2/dataset/drone/250312_sejong/fasterrcnn_inference_cropped3', help='Directory to save visualization results')
    parser.add_argument('--num-samples', type=int, default=-1, help='Number of samples to process (-1 for all)')
    parser.add_argument('--score-threshold', type=float, default=0.5, help='Score threshold for visualization and saving')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--output-json', default='fasterrcnn_coco_predictions.json', help='Path to save the COCO format prediction JSON file')
    parser.add_argument('--no-vis-crop', action='store_true', help='Disable visual cropping of the output images')
    
    args = parser.parse_args()
    
    wandb.init(project='DELIVER', name=f'inference_rgb_{Path(args.checkpoint).stem}', 
               tags=['inference', 'visualization', 'prediction_save', 'rgb_only'], config=vars(args))
    
    visualizer = RGBInferenceVisualizer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    visualizer.run_inference_and_visualize(
        output_dir=args.output_dir, 
        num_samples=args.num_samples, 
        score_threshold=args.score_threshold, 
        output_json_path=args.output_json,
        use_vis_crop=not args.no_vis_crop
    )
    wandb.finish()

if __name__ == '__main__':
    main()