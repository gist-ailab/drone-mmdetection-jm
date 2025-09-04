#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import colorsys 
import mmengine
from mmengine.registry import init_default_scope, METRICS
from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from typing import Tuple, Dict, List, Optional
import wandb
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

# 사용자의 커스텀 모듈을 레지스트리에 등록 (필요 시)
# from mcdet import *

class RGBDetectorVisualizer:
    """
    RGB 이미지 전용 CocoDataset에 대한 추론, mAP 계산, 시각화를 수행하는 클래스.
    """
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.cfg = mmengine.Config.fromfile(config_path)
        init_default_scope(self.cfg.get('default_scope', 'mmdet'))
        self.model = init_detector(config_path, checkpoint_path, device=device, cfg_options={'strict': False})
        
        self.classes = self.cfg.val_dataloader.dataset.metainfo['classes']
        self.colors = self.cfg.val_dataloader.dataset.metainfo.get('palette')
        if self.colors is None:
            print("Warning: Palette not found in config, generating random colors.")
            np.random.seed(42)
            self.colors = np.random.randint(0, 256, size=(len(self.classes), 3)).tolist()

    def crop_and_adjust_bboxes(
        self, boxes: np.ndarray, labels: np.ndarray, crop_box: List[int],
        scores: Optional[np.ndarray] = None
    ) -> Tuple:
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
            final_scores = scores[valid_indices]
            return final_boxes, final_labels, final_scores
        else:
            return final_boxes, final_labels

    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                   text_prefix: str, box_format: str = 'xyxy') -> np.ndarray:
        output_img = image.copy()
        for bbox, label in zip(boxes, labels):
            if box_format == 'xywh':
                x, y, w, h = bbox.astype(int)
                x1, y1, x2, y2 = x, y, x + w, y + h
            else: # xyxy
                x1, y1, x2, y2 = bbox.astype(int)

            class_name = self.classes[label]
            color = self.colors[label % len(self.colors)]
            label_text = f'{text_prefix}: {class_name}'
            
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(output_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return output_img
    
    # ✨ 최적화된 함수
    def draw_translucent_boxes(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                               text_prefix: str, box_format: str = 'xyxy', alpha: float = 0.5) -> np.ndarray:
        output_img = image.copy()
        overlay = output_img.copy()

        draw_info = []
        for bbox, label in zip(boxes, labels):
            if box_format == 'xywh':
                x, y, w, h = bbox.astype(int)
                x1, y1, x2, y2 = x, y, x + w, y + h
            else: # xyxy
                x1, y1, x2, y2 = bbox.astype(int)
            
            r, g, b = self.colors[label % len(self.colors)]
            hls = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
            lightness = min(1.0, hls[1] * 1.3)
            new_rgb = colorsys.hls_to_rgb(hls[0], lightness, hls[2])
            gt_color = (int(new_rgb[0] * 255), int(new_rgb[1] * 255), int(new_rgb[2] * 255))
            label_text = f'{text_prefix}: {self.classes[label]}'
            draw_info.append({'coords': (x1, y1, x2, y2), 'color': gt_color, 'text': label_text})

        for info in draw_info:
            x1, y1, x2, y2 = info['coords']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), info['color'], -1)

        cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0, output_img)

        for info in draw_info:
            x1, y1, x2, y2 = info['coords']
            gt_color = info['color']
            label_text = info['text']
            cv2.rectangle(output_img, (x1, y1), (x2, y2), gt_color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_img, (x1, y1 - text_h - 5), (x1 + text_w, y1), gt_color, -1)
            cv2.putText(output_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return output_img

    @torch.no_grad()
    def visualize_from_validation_set(self, output_dir: str, num_samples: int, score_threshold: float, use_crop: bool, eval_only: bool):
        if not eval_only:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'w_gt'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'wo_gt'), exist_ok=True)
        
        val_dataset_cfg = self.cfg.val_dataloader.dataset
        dataset = DATASETS.build(val_dataset_cfg)
        
        total_samples = len(dataset) if num_samples < 0 else min(num_samples, len(dataset))

        print(f"'{val_dataset_cfg.type}' 데이터셋에서 {len(dataset)}개의 샘플을 찾았습니다.")
        if eval_only:
             print(f"총 {total_samples}개의 샘플에 대해 평가(mAP 계산)만 진행합니다...")
        else:
            print(f"총 {total_samples}개의 샘플에 대해 시각화 및 평가를 진행합니다...")
        
        crop_box = [40, 110, 480, 480] if use_crop else None
        if crop_box:
            print(f"모든 Bbox는 다음 영역으로 잘립니다: {crop_box}")

        self.model.eval()
        all_predictions_for_eval = []

        for i, data