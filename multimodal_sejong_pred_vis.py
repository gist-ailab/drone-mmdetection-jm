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

# 사용자의 커스텀 모듈 등록
from mcdet import *

class SejongMultimodalVisualizer:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.cfg = mmengine.Config.fromfile(config_path)
        mmengine.registry.init_default_scope(self.cfg.get('default_scope', 'mmdet'))
        self.model = init_detector(config_path, checkpoint_path, device=device, cfg_options={'strict': False})
        self.classes = self.cfg.val_dataloader.dataset.metainfo.classes
        self.colors = self.cfg.val_dataloader.dataset.metainfo.palette

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
        # 좌표계를 Crop된 이미지 기준으로 변환
        final_boxes[:, [0, 2]] -= crop_x1
        final_boxes[:, [1, 3]] -= crop_y1

        if scores is not None:
            return final_boxes, final_labels, scores[valid_indices]
        else:
            return final_boxes, final_labels

    # --- (load_multimodal_images, create_multimodal_concat, draw_boxes, draw_translucent_boxes, add_modality_labels 함수는 이전과 동일하게 유지) ---
    # (코드 간결성을 위해 여기에 다시 첨부하지는 않겠습니다. 이전 코드의 함수들을 그대로 사용하시면 됩니다.)

    def load_multimodal_images(self, rgb_img_path: str) -> Dict[str, np.ndarray]:
        p = Path(rgb_img_path)
        modality_paths = {
            'rgb': str(p),
            'depth': str(p).replace('group_rgb', 'group_depth'),
            'event': str(p).replace('group_rgb', 'group_ir'),
            'lidar': str(p).replace('group_rgb', 'group_intensity')
        }
        images = {}
        for modality, path in modality_paths.items():
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None: continue
                if len(img.shape) == 2:
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                images[modality] = img
        return images

    def create_multimodal_concat(self, images: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
        if not images: raise ValueError("사용 가능한 이미지가 없습니다.")
        ref_img = images.get('rgb')
        if ref_img is None: ref_img = next(iter(images.values()))
        h, w = ref_img.shape[:2]
        grid_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        modality_positions = {'rgb': (0, 0), 'depth': (0, 1), 'event': (1, 0), 'lidar': (1, 1)}
        for modality, (row, col) in modality_positions.items():
            if modality in images:
                img = images[modality]
                if img.shape[:2] != (h, w): img = cv2.resize(img, (w, h))
                if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if modality == 'rgb':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                grid_img[row*h:(row+1)*h, col*w:(col+1)*w] = img
        return grid_img, (h, w)
    
    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                   img_shape: Tuple[int, int], text_prefix: str,
                   box_format: str = 'xyxy',
                   draw_on: List[str] = None) -> np.ndarray:
        h, w = img_shape
        modality_offsets = {'rgb': (0, 0), 'depth': (0, w), 'event': (h, 0), 'lidar': (h, w)}
        target_modalities = draw_on if draw_on else modality_offsets.keys()
        for bbox, label in zip(boxes, labels):
            if box_format == 'xywh':
                x, y, box_w, box_h = bbox.astype(int); x1, y1, x2, y2 = x, y, x + box_w, y + box_h
            else: x1, y1, x2, y2 = bbox.astype(int)
            class_name = self.classes[label]; color = self.colors[label % len(self.colors)]
            label_text = f'{text_prefix}: {class_name}'
            for modality in target_modalities:
                if modality not in modality_offsets: continue
                offset_y, offset_x = modality_offsets[modality]
                x1_s, y1_s, x2_s, y2_s = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
                cv2.rectangle(image, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1_s, y1_s - text_h - 5), (x1_s + text_w, y1_s), color, -1)
                cv2.putText(image, label_text, (x1_s, y1_s - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return image
    
    def draw_translucent_boxes(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                               img_shape: Tuple[int, int], text_prefix: str,
                               box_format: str = 'xyxy', draw_on: List[str] = None, alpha: float = 0.5) -> np.ndarray:
        output_img = image.copy(); overlay = output_img.copy()
        h, w = img_shape; modality_offsets = {'rgb': (0, 0), 'depth': (0, w), 'event': (h, 0), 'lidar': (h, w)}
        target_modalities = draw_on if draw_on else modality_offsets.keys()
        draw_info = []
        for bbox, label in zip(boxes, labels):
            if box_format == 'xywh': x, y, box_w, box_h = bbox.astype(int); x1, y1, x2, y2 = x, y, x + box_w, y + box_h
            else: x1, y1, x2, y2 = bbox.astype(int)
            r, g, b = self.colors[label % len(self.colors)]; hls = colorsys.rgb_to_hls(r / 255., g / 255., b / 255.)
            lightness = min(1.0, hls[1] * 1.3); new_rgb = colorsys.hls_to_rgb(hls[0], lightness, hls[2])
            gt_color = (int(new_rgb[0] * 255), int(new_rgb[1] * 255), int(new_rgb[2] * 255))
            label_text = f'{text_prefix}: {self.classes[label]}'
            draw_info.append({'coords': (x1, y1, x2, y2), 'color': gt_color, 'text': label_text})
        for info in draw_info:
            x1, y1, x2, y2 = info['coords']
            for modality in target_modalities:
                if modality not in modality_offsets: continue
                offset_y, offset_x = modality_offsets[modality]
                x1_s, y1_s, x2_s, y2_s = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
                cv2.rectangle(overlay, (x1_s, y1_s), (x2_s, y2_s), info['color'], -1)
        cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0, output_img)
        for info in draw_info:
            x1, y1, x2, y2 = info['coords']; gt_color = info['color']; label_text = info['text']
            for modality in target_modalities:
                if modality not in modality_offsets: continue
                offset_y, offset_x = modality_offsets[modality]
                x1_s, y1_s, x2_s, y2_s = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
                cv2.rectangle(output_img, (x1_s, y1_s), (x2_s, y2_s), gt_color, 2)
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output_img, (x1_s, y1_s - text_h - 5), (x1_s + text_w, y1_s), gt_color, -1)
                cv2.putText(output_img, label_text, (x1_s, y1_s - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return output_img

    def add_modality_labels(self, concat_img: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        h, w = img_shape
        modality_labels = {'RGB': (20, 30), 'Depth': (w + 20, 30), 'IR': (20, h + 30), 'LiDAR': (w + 20, h + 30)}
        for label, (x, y) in modality_labels.items():
            cv2.putText(concat_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return concat_img
    
    @torch.no_grad()
    def run_inference_and_visualize(self, output_dir: str, num_samples: int, score_threshold: float,
                                      draw_on_modalities: List[str], output_json_path: str):
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'w_gt'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'wo_gt'), exist_ok=True)
        
        val_dataset_cfg = self.cfg.val_dataloader.dataset
        dataset = DATASETS.build(val_dataset_cfg)
        dataset.full_init()
        
        total_samples = len(dataset) if num_samples < 0 else min(num_samples, len(dataset))

        print(f"'{val_dataset_cfg.type}' 데이터셋에서 {len(dataset)}개의 샘플을 찾았습니다.")
        print(f"총 {total_samples}개의 샘플에 대해 시각화 및 예측 저장을 진행합니다...")

        crop_box = [40, 110, 480, 480]
        print(f"모든 시각화 Bbox는 다음 영역으로 잘립니다: {crop_box}")

        self.model.eval()
        coco_results = []

        for i, data in enumerate(dataset):
            if i >= total_samples: break
            
            original_data_sample = data['data_samples']
            image_id = original_data_sample.img_id
            
            batched_data = { 'inputs': [[item] for item in data['inputs']], 'data_samples': [original_data_sample] }
            processed_data = self.model.data_preprocessor(batched_data, training=False)
            predictions = self.model.forward(**processed_data, mode='predict')
            pred_sample = predictions[0]
            
            scale_factor = processed_data['data_samples'][0].metainfo['scale_factor']
            pred_instances = pred_sample.pred_instances[pred_sample.pred_instances.scores > score_threshold]
            
            pred_boxes_xyxy = pred_instances.bboxes.cpu().numpy() / np.tile(scale_factor, 2)
            pred_labels = pred_instances.labels.cpu().numpy()
            pred_scores = pred_instances.scores.cpu().numpy()

            for box, label, score in zip(pred_boxes_xyxy, pred_labels, pred_scores):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                category_id = dataset.cat_ids[label]
                coco_results.append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [float(coord) for coord in [x1, y1, w, h]], # float으로 변환
                    'score': float(score)
                })
            
            # --- 시각화 수행 ---
            gt_instances = original_data_sample.gt_instances
            
            # ✨ 핵심 수정: gt_instances에 bbox가 있는지 확인합니다.
            if 'bboxes' in gt_instances:
                gt_boxes_xyxy = gt_instances.bboxes.cpu().numpy()
                gt_boxes_xyxy[:, 2:] += gt_boxes_xyxy[:, :2]
                gt_labels = gt_instances.labels.cpu().numpy()
            else:
                # GT가 없는 경우, 빈 배열을 생성합니다.
                gt_boxes_xyxy = np.empty((0, 4))
                gt_labels = np.empty((0,))

            vis_gt_boxes, vis_gt_labels = self.crop_and_adjust_bboxes(gt_boxes_xyxy, gt_labels, crop_box)
            vis_pred_boxes, vis_pred_labels, _ = self.crop_and_adjust_bboxes(
                pred_boxes_xyxy, pred_labels, crop_box, scores=pred_scores)
            
            rgb_img_path = original_data_sample.img_path[0]
            img_id_str = Path(rgb_img_path).stem
            print(f"[{i+1}/{total_samples}] 처리 중: {img_id_str}")

            images = self.load_multimodal_images(rgb_img_path)
            if not images: continue

            cr_x1, cr_y1, cr_x2, cr_y2 = crop_box
            cropped_images = {}
            for modality, img in images.items():
                h_orig, w_orig = img.shape[:2]
                if (h_orig, w_orig) != (640, 480): img = cv2.resize(img, (480, 640))
                cropped_images[modality] = img[cr_y1:cr_y2, cr_x1:cr_x2]
            
            concat_img, img_shape = self.create_multimodal_concat(cropped_images)
            result_img = self.draw_boxes(concat_img.copy(), vis_pred_boxes, vis_pred_labels, img_shape, 
                                         text_prefix="Pred", box_format='xyxy', draw_on=draw_on_modalities)
            result_img = self.add_modality_labels(result_img, img_shape)
            cv2.imwrite(os.path.join(output_dir, 'wo_gt', f'{img_id_str}.jpg'), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            
            result_img_w_gt = self.draw_translucent_boxes(result_img, vis_gt_boxes, vis_gt_labels, img_shape, 
                                                          text_prefix="GT", box_format='xyxy', draw_on=draw_on_modalities, alpha=0.5)
            cv2.imwrite(os.path.join(output_dir, 'w_gt', f'{img_id_str}.jpg'), cv2.cvtColor(result_img_w_gt, cv2.COLOR_RGB2BGR))

        print(f"\n총 {len(coco_results)}개의 예측 결과를 '{output_json_path}' 파일에 저장합니다.")
        with open(output_json_path, 'w') as f:
            json.dump(coco_results, f, indent=4)
        
        print("\n시각화 및 예측 저장이 완료되었습니다.")


def main():
    parser = argparse.ArgumentParser(description='Sejong Multimodal Detection Visualization and Prediction Saving')
    parser.add_argument(
        '--config', 
        default='/SSDb/jemo_maeng/src/Project/Drone/detection/drone-mmdetection-jm/work_dirs/sejong2504_cmnextp_b2_rcnn_multiscale_v2/yeon-sejong2504_cmnextp_rcnn_lr0.01_ep50_v2.py',
        help='Model config file path'
    )
    parser.add_argument(
        '--checkpoint',
        default='/SSDb/jemo_maeng/src/Project/Drone/detection/drone-mmdetection-jm/work_dirs/sejong2504_cmnextp_b2_rcnn_multiscale_v2/best_coco_bbox_mAP_epoch_30.pth',
        help='Model checkpoint file path'
    )
    
    parser.add_argument('--output-dir', default='outputs/multimodal_results', help='Directory to save visualization results')
    parser.add_argument('--num-samples', type=int, default=-1, help='Number of samples to process (-1 for all)')
    parser.add_argument('--score-threshold', type=float, default=0.5, help='Score threshold for visualization and saving')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--draw-on', nargs='+', default=['rgb'], help='Modalities to draw boxes on (e.g., rgb depth)')
    parser.add_argument('--output-json', default='coco_predictions.json', help='Path to save the COCO format prediction JSON file')
    
    args = parser.parse_args()
    
    wandb.init(project='DELIVER', name=f'inference_{Path(args.checkpoint).stem}', 
               tags=['inference', 'visualization', 'prediction_save', 'cmnext'], config=vars(args))
    
    visualizer = SejongMultimodalVisualizer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    visualizer.run_inference_and_visualize(
        output_dir=args.output_dir, 
        num_samples=args.num_samples, 
        score_threshold=args.score_threshold, 
        draw_on_modalities=args.draw_on,
        output_json_path=args.output_json # ✨ 인자 전달
    )
    wandb.finish()

if __name__ == '__main__':
    main()