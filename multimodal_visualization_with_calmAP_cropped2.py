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
import tempfile
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import DATASETS
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import wandb
from typing import Tuple, Dict, List, Optional

# 사용자의 커스텀 모듈 등록
from mcdet import *

class MultimodalEvaluator:
    """
    멀티모달 데이터에 대한 추론, 시각화, 안정적인 mAP 평가를 모두 수행하는 통합 클래스.
    """
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.cfg = mmengine.Config.fromfile(config_path)
        # init_default_scope는 inference_detector/init_detector에서 처리되므로 불필요
        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.classes = self.cfg.val_dataloader.dataset.metainfo.classes
        self.colors = self.cfg.val_dataloader.dataset.metainfo.palette

    # ▼▼▼ 시각화에 필요한 헬퍼 함수들 (기존 코드와 동일) ▼▼▼
    def crop_and_adjust_bboxes(self, boxes: np.ndarray, labels: np.ndarray, crop_box: List[int], scores=None):
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
        return final_boxes, final_labels

    def load_multimodal_images(self, rgb_img_path: str):
        p = Path(rgb_img_path)
        modality_paths = {'rgb': str(p), 'depth': str(p).replace('group_rgb', 'group_depth'), 'event': str(p).replace('group_rgb', 'group_ir'), 'lidar': str(p).replace('group_rgb', 'group_intensity')}
        images = {}
        for modality, path in modality_paths.items():
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None: continue
                if len(img.shape) == 2: img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                images[modality] = img
        return images

    def create_multimodal_concat(self, images: Dict[str, np.ndarray]):
        ref_img = images.get('rgb') or next(iter(images.values()))
        h, w = ref_img.shape[:2]
        grid_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        modality_positions = {'rgb': (0, 0), 'depth': (0, 1), 'event': (1, 0), 'lidar': (1, 1)}
        for modality, (row, col) in modality_positions.items():
            if modality in images:
                img = images[modality]
                if img.shape[:2] != (h, w): img = cv2.resize(img, (w, h))
                if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if modality == 'rgb': img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                grid_img[row*h:(row+1)*h, col*w:(col+1)*w] = img
        return grid_img, (h, w)

    def draw_boxes(self, image, boxes, labels, img_shape, text_prefix, box_format='xyxy', draw_on=None):
        h, w = img_shape
        modality_offsets = {'rgb': (0, 0), 'depth': (0, w), 'event': (h, 0), 'lidar': (h, w)}
        target_modalities = draw_on if draw_on else modality_offsets.keys()
        for bbox, label in zip(boxes, labels):
            x1, y1, x2, y2 = bbox.astype(int)
            color = self.colors[label % len(self.colors)]
            label_text = f'{text_prefix}: {self.classes[label]}'
            for modality in target_modalities:
                if modality not in modality_offsets: continue
                offset_y, offset_x = modality_offsets[modality]
                x1_s, y1_s, x2_s, y2_s = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
                cv2.rectangle(image, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1_s, y1_s - text_h - 5), (x1_s + text_w, y1_s), color, -1)
                cv2.putText(image, label_text, (x1_s, y1_s - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return image

    def draw_translucent_boxes(self, image, boxes, labels, img_shape, text_prefix, box_format='xyxy', draw_on=None, alpha=0.5):
        # ... (이전과 동일한 반투명 박스 그리기 로직) ...
        return image # 실제 구현은 이전 코드 참고

    def add_modality_labels(self, image, img_shape):
        # ... (이전과 동일한 라벨 추가 로직) ...
        return image

    # ▼▼▼ 핵심 로직: 추론, 시각화, 예측 저장을 한번에 수행 ▼▼▼
    def run(self, args):
        """메인 실행 함수"""
        if not args.eval_only:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, 'w_gt'), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, 'wo_gt'), exist_ok=True)

        # 1. 데이터셋 빌드 (추론할 이미지 목록 가져오기 위함)
        dataset_cfg = self.cfg.val_dataloader.dataset
        dataset_cfg.ann_file = args.ann_file
        # 파이프라인은 최소한으로 유지하여 원본 이미지 경로만 가져옴
        dataset_cfg.pipeline = [dict(type='LoadAnnotations', with_bbox=True)]
        dataset = DATASETS.build(dataset_cfg)
        dataset.full_init()

        total_samples = len(dataset) if args.num_samples < 0 else min(args.num_samples, len(dataset))
        print(f"'{dataset_cfg.type}'에서 총 {total_samples}개 샘플을 처리합니다.")

        coco_gt = COCO(args.ann_file)
        coco_predictions = []
        progress_bar = mmengine.ProgressBar(total_samples)

        for i in range(total_samples):
            data = dataset[i]
            img_id = data['img_id']
            # 멀티모달 데이터셋은 img_path가 리스트일 수 있음
            img_path = data['img_path'][0] if isinstance(data['img_path'], list) else data['img_path']

            # 모델 추론
            result = inference_detector(self.model, img_path)
            
            # 예측 결과 필터링 및 COCO 형식으로 저장
            pred_instances = result.pred_instances[result.pred_instances.scores > args.score_threshold]
            for bbox, label, score in zip(pred_instances.bboxes, pred_instances.labels, pred_instances.scores):
                x1, y1, x2, y2 = bbox.cpu().numpy().tolist()
                w, h = x2 - x1, y2 - y1
                coco_predictions.append({
                    'image_id': img_id,
                    'category_id': dataset.cat_ids[label.cpu().item()],
                    'bbox': [x1, y1, w, h],
                    'score': float(score.cpu().item())
                })
            
            # 시각화 로직 (`--eval-only`가 아닐 때만 실행)
            if not args.eval_only:
                # ... (이전 멀티모달 스크립트의 복잡한 시각화 로직을 여기에 그대로 적용)
                # 예: crop_box 정의, load_multimodal_images, create_multimodal_concat, draw_boxes 등
                pass # 여기에 시각화 로직 구현

            progress_bar.update()

        # 2. 예측 결과를 JSON 파일로 저장
        pred_ann_file = os.path.join(args.output_dir, 'predictions.json')
        with open(pred_ann_file, 'w') as f:
            json.dump(coco_predictions, f)
        print(f"\n모든 예측 결과를 '{pred_ann_file}'에 저장했습니다.")

        # 3. pycocotools로 평가
        self.evaluate(args.ann_file, pred_ann_file)

    def evaluate(self, gt_ann_file, pred_ann_file):
        """두 파일을 비교하여 mAP를 계산하고 출력"""
        coco_gt = COCO(gt_ann_file)
        coco_dt = coco_gt.loadRes(pred_ann_file)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        print("\n---*--- COCO 평가 결과 ---*---")
        coco_eval.summarize()
        print("---*--------------------*---")

        if wandb.run:
            stats = coco_eval.stats
            wandb.log({
                'mAP/mAP': stats[0], 'mAP/mAP_50': stats[1], 'mAP/mAP_75': stats[2],
                'mAP/mAP_s': stats[3], 'mAP/mAP_m': stats[4], 'mAP/mAP_l': stats[5]
            })
            print("\n평가 지표를 wandb에 로깅했습니다.")


def main():
    parser = argparse.ArgumentParser(description='Multimodal Detection Inference and Evaluation')
    # --- 기존 인자들과 기본값 유지 ---
    parser.add_argument('--config', default='/SSDb/jemo_maeng/src/Project/Drone/detection/drone-mmdetection-jm/work_dirs/sejong2504_cmnextp_b2_rcnn_multiscale_v2/yeon-sejong2504_cmnextp_rcnn_lr0.01_ep50_v2.py', help='모델 config 파일 경로')
    parser.add_argument('--checkpoint', default='/SSDb/jemo_maeng/src/Project/Drone/detection/drone-mmdetection-jm/work_dirs/sejong2504_cmnextp_b2_rcnn_multiscale_v2/best_coco_bbox_mAP_epoch_30.pth', help='모델 weight 파일 경로')
    parser.add_argument('--output-dir', default='/ailab_mat2/dataset/drone/250312_sejong/cmnextp_inference_ep30_cropped2', help='시각화 및 예측 결과 저장 디렉토리')
    parser.add_argument('--num-samples', type=int, default=-1, help='처리할 샘플 수 (-1이면 전체)')
    parser.add_argument('--score-threshold', type=float, default=0.4, help='신뢰도 임계값')
    parser.add_argument('--device', default='cuda:0', help='사용할 디바이스')
    parser.add_argument('--draw-on', nargs='+', default=['rgb'], help='박스를 그릴 모달리티 지정')
    parser.add_argument('--eval-only', action='store_true', help='이미지 저장을 생략하고 mAP 평가만 수행')
    
    # --- 평가를 위한 새로운 필수 인자 ---
    parser.add_argument('--ann-file', default='/ailab_mat2/dataset/drone/250312_sejong/drone_250312_sejong_multimodal_coco/labels/test_filtered.json', help='(필수) 평가의 기준이 될 Ground Truth annotation 파일 경로')
    
    args = parser.parse_args()

    wandb.init(project='DELIVER', name=f'eval_{"eval_only_" if args.eval_only else ""}{Path(args.checkpoint).stem}', config=vars(args))

    evaluator = MultimodalEvaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    evaluator.run(args)
    
    wandb.finish()

if __name__ == '__main__':
    main()