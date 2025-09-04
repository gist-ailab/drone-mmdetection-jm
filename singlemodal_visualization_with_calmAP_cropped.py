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
    def visualize_from_validation_set(self, output_dir: str, num_samples: int, score_threshold: float,
                                      draw_on_modalities: List[str], eval_only: bool):
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

        crop_box = [40, 110, 480, 480]
        print(f"모든 Bbox는 다음 영역으로 잘립니다: {crop_box}")

        self.model.eval()
        
        # ✨ 1. Evaluator를 루프 시작 전에 미리 생성
        evaluator_cfg = self.cfg.val_evaluator
        evaluator = METRICS.build(evaluator_cfg)
        evaluator.dataset_meta = dataset.metainfo

        for i, data in enumerate(dataset):
            if i >= total_samples: break
            
            original_data_sample = data['data_samples']
            batched_data = { 'inputs': [[item] for item in data['inputs']], 'data_samples': [original_data_sample] }
            processed_data = self.model.data_preprocessor(batched_data, training=False)
            predictions = self.model.forward(**processed_data, mode='predict')
            pred_sample = predictions[0]

            scale_factor = processed_data['data_samples'][0].metainfo['scale_factor']
            gt_instances = original_data_sample.gt_instances
            gt_boxes_wh = gt_instances.bboxes.cpu().numpy()
            gt_labels = gt_instances.labels.cpu().numpy()
            gt_boxes_xyxy = gt_boxes_wh.copy()
            gt_boxes_xyxy[:, 2] += gt_boxes_xyxy[:, 0]
            gt_boxes_xyxy[:, 3] += gt_boxes_xyxy[:, 1]
            
            pred_instances = pred_sample.pred_instances[pred_sample.pred_instances.scores > score_threshold]
            pred_boxes_scaled = pred_instances.bboxes.cpu().numpy()
            pred_labels = pred_instances.labels.cpu().numpy()
            pred_scores = pred_instances.scores.cpu().numpy()
            if pred_boxes_scaled.shape[0] > 0:
                rescale_factor = np.tile(scale_factor, 2)
                pred_boxes_xyxy = pred_boxes_scaled / rescale_factor
            else:
                pred_boxes_xyxy = np.empty((0, 4))

            cropped_gt_boxes, cropped_gt_labels = self.crop_and_adjust_bboxes(gt_boxes_xyxy, gt_labels, crop_box)
            cropped_pred_boxes, cropped_pred_labels, cropped_pred_scores = self.crop_and_adjust_bboxes(
                pred_boxes_xyxy, pred_labels, crop_box, scores=pred_scores)
            
            eval_pred_sample = DetDataSample()
            eval_pred_sample.pred_instances = InstanceData(
                bboxes=torch.from_numpy(cropped_pred_boxes),
                labels=torch.from_numpy(cropped_pred_labels),
                scores=torch.from_numpy(cropped_pred_scores))
            eval_pred_sample.gt_instances = InstanceData(
                bboxes=torch.from_numpy(cropped_gt_boxes),
                labels=torch.from_numpy(cropped_gt_labels))
            
            # ✨ 2. 리스트에 추가하는 대신, evaluator.process()를 호출
            # data_batch는 dataloader에서 나온 원본 data, data_samples는 처리된 예측 결과 리스트
            evaluator.process(data_batch=data, data_samples=[eval_pred_sample.to_dict()])

            if not eval_only:
                rgb_img_path = original_data_sample.img_path[0]
                img_id = Path(rgb_img_path).stem
                print(f"[{i+1}/{total_samples}] 시각화 처리 중: {img_id}")
                # ... (이하 시각화 및 이미지 저장 로직은 동일) ...
                images = self.load_multimodal_images(rgb_img_path)
                if not images: continue
                cr_x1, cr_y1, cr_x2, cr_y2 = crop_box
                cropped_images = {}
                for modality, img in images.items():
                    h_orig, w_orig = img.shape[:2];
                    if (h_orig, w_orig) != (640, 480): img = cv2.resize(img, (480, 640))
                    cropped_images[modality] = img[cr_y1:cr_y2, cr_x1:cr_x2]
                concat_img, img_shape = self.create_multimodal_concat(cropped_images)
                result_img = self.draw_boxes(concat_img.copy(), cropped_pred_boxes, cropped_pred_labels, img_shape, 
                                             text_prefix="Pred", box_format='xyxy', draw_on=draw_on_modalities)
                result_img = self.add_modality_labels(result_img, img_shape)
                output_path_wogt = os.path.join(output_dir, 'wo_gt', f'{img_id}.jpg')
                cv2.imwrite(output_path_wogt, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                result_img_w_gt = self.draw_translucent_boxes(result_img, cropped_gt_boxes, cropped_gt_labels, img_shape, 
                                                              text_prefix="GT", box_format='xyxy', 
                                                              draw_on=draw_on_modalities, alpha=0.5)
                output_path_wgt = os.path.join(output_dir, 'w_gt', f'{img_id}.jpg')
                cv2.imwrite(output_path_wgt, cv2.cvtColor(result_img_w_gt, cv2.COLOR_RGB2BGR))
            else:
                if (i + 1) % 100 == 0 or (i + 1) == total_samples:
                    print(f"[{i+1}/{total_samples}] 평가 처리 중...")
        
        if not eval_only:
            print(f"\n시각화 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")
        
        # ✨ 3. 모든 샘플 처리가 끝난 후, evaluate()를 호출하여 최종 결과를 계산
        print(f"\nCrop된 Bbox 기준, {total_samples}개 샘플에 대한 평가(mAP)를 시작합니다...")
        eval_metrics = evaluator.evaluate(size=total_samples)

        print("\n---*--- 평가 결과 (Cropped) ---*---")
        for key, value in eval_metrics.items():
            if isinstance(value, np.float32): eval_metrics[key] = round(float(value), 4)
            print(f"{key:<25}: {eval_metrics[key]}")
        print("---*---------------------------*---")

        if wandb.run:
            wandb_metrics = {f'eval_cropped/{k}': v for k, v in eval_metrics.items()}
            wandb.log(wandb_metrics)
            print("\nCrop 기준 평가 지표를 wandb에 성공적으로 로깅했습니다.")

def main():
    parser = argparse.ArgumentParser(description='RGB CocoDataset Detection Visualization and Evaluation')
    parser.add_argument('--config', default='custom_configs/DELIVER/lecun-sejong2504_cmnext_rcnn_v2.py', help='모델 config 파일 경로')
    parser.add_argument('--checkpoint', default='work_dirs/sejong2504_faster_rcnn__v2/best_coco_bbox_mAP_epoch_40.pth', help='모델 weight 파일 경로')
    parser.add_argument('--output-dir', default='outputs/rgb_inference_results', help='시각화 결과 저장 디렉토리')
    parser.add_argument('--num-samples', type=int, default=-1, help='시각화 및 평가할 샘플 수 (-1이면 전체)')
    parser.add_argument('--score-threshold', type=float, default=0.3, help='신뢰도 임계값')
    parser.add_argument('--device', default='cuda:0', help='사용할 디바이스')
    parser.add_argument('--no-crop', action='store_true', help='이미지와 Bbox를 Crop하지 않음')
    # ✨ --eval-only 인자 추가
    parser.add_argument('--eval-only', action='store_true', help='이미지 저장을 생략하고 mAP 평가만 수행합니다.')
    
    args = parser.parse_args()
    
    # ✨ wandb 실행 이름 및 태그에 모드 반영
    run_mode = 'eval_only' if args.eval_only else 'visualize'
    crop_mode = 'full' if args.no_crop else 'cropped'
    wandb_run_name = f'{run_mode}_{crop_mode}_{Path(args.checkpoint).stem}'
    wandb.init(
        project='DELIVER', 
        name=wandb_run_name, 
        tags=['inference', run_mode, 'evaluation', 'rgb_only', crop_mode], 
        config=vars(args)
    )
    
    visualizer = RGBDetectorVisualizer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    visualizer.visualize_from_validation_set(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        score_threshold=args.score_threshold,
        use_crop=not args.no_crop,
        eval_only=args.eval_only # ✨ 인자 전달
    )
    wandb.finish()

if __name__ == '__main__':
    main()