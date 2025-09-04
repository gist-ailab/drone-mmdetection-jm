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
from mmengine.registry import init_default_scope, METRICS
from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from typing import Tuple, Dict, List, Optional
import wandb
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

# ✨ pycocotools API를 직접 사용하기 위해 import
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 사용자의 커스텀 모듈 등록
from mcdet import *

class SejongMultimodalVisualizer:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.cfg = mmengine.Config.fromfile(config_path)
        init_default_scope(self.cfg.get('default_scope', 'mmdet'))
        self.model = init_detector(config_path, checkpoint_path, device=device, cfg_options={'strict': False})
        self.classes = self.cfg.val_dataloader.dataset.metainfo.classes
        self.colors = self.cfg.val_dataloader.dataset.metainfo.palette

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
        
        # ✨ 1. MMEngine Evaluator 대신 COCO 표준 형식의 예측 결과를 담을 리스트를 생성합니다.
        coco_predictions = []

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

            # ✨ 2. 루프 내에서 예측 결과를 COCO JSON 형식으로 변환하여 리스트에 추가합니다.
            image_id = original_data_sample.img_id
            for bbox, label, score in zip(cropped_pred_boxes, cropped_pred_labels, cropped_pred_scores):
                # bbox: xyxy -> xywh
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                # label: class index -> coco category id
                category_id = dataset.cat_ids[label]
                
                coco_predictions.append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [x1, y1, w, h],
                    'score': float(score)
                })

            if not eval_only:
                # ... (시각화 로직은 변경 없음) ...
                rgb_img_path = original_data_sample.img_path[0]; img_id = Path(rgb_img_path).stem
                print(f"[{i+1}/{total_samples}] 시각화 처리 중: {img_id}")
                images = self.load_multimodal_images(rgb_img_path)
                if not images: continue
                cr_x1, cr_y1, cr_x2, cr_y2 = crop_box; cropped_images = {}
                for modality, img in images.items():
                    h_orig, w_orig = img.shape[:2]
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
        
        # ✨ 3. 루프 종료 후, pycocotools를 사용하여 mAP를 직접 계산합니다.
        print(f"\nCrop된 Bbox 기준, {total_samples}개 샘플에 대한 평가(mAP)를 시작합니다...")
        
        if not coco_predictions:
            print("평가할 예측 결과가 없습니다.")
            return

        # Ground Truth 로드
        coco_gt = COCO(dataset.ann_file)
        # 예측 결과를 임시 파일에 저장하고 로드
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as tmp_file:
            json.dump(coco_predictions, tmp_file)
            tmp_file.flush()
            coco_dt = coco_gt.loadRes(tmp_file.name)

        # COCOeval 객체 생성 및 평가 수행
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        
        # Crop된 GT만 평가 대상으로 삼기 위해 이미지 ID 필터링
        img_ids_to_eval = sorted(list(coco_gt.imgs.keys()))
        coco_eval.params.imgIds = img_ids_to_eval[:total_samples]

        coco_eval.evaluate()
        coco_eval.accumulate()
        print("\n---*--- 평가 결과 (Cropped, pycocotools) ---*---")
        coco_eval.summarize()
        print("---*------------------------------------*---")
        
        if wandb.run:
            stats = coco_eval.stats
            wandb_metrics = {
                'eval_cropped/mAP': stats[0],
                'eval_cropped/mAP_50': stats[1],
                'eval_cropped/mAP_75': stats[2],
                'eval_cropped/mAP_s': stats[3],
                'eval_cropped/mAP_m': stats[4],
                'eval_cropped/mAP_l': stats[5],
            }
            wandb.log(wandb_metrics)
            print("\nCrop 기준 평가 지표를 wandb에 성공적으로 로깅했습니다.")

def main():
    # argparse 부분은 이전과 동일하므로 생략합니다.
    parser = argparse.ArgumentParser(description='Sejong Multimodal Detection Visualization and Evaluation with Cropping')
    parser.add_argument('--config', default='/SSDb/jemo_maeng/src/Project/Drone/detection/drone-mmdetection-jm/work_dirs/sejong2504_cmnextp_b2_rcnn_multiscale_v2/yeon-sejong2504_cmnextp_rcnn_lr0.01_ep50_v2.py', help='모델 config 파일 경로')
    parser.add_argument('--checkpoint', default='/SSDb/jemo_maeng/src/Project/Drone/detection/drone-mmdetection-jm/work_dirs/sejong2504_cmnextp_b2_rcnn_multiscale_v2/best_coco_bbox_mAP_epoch_30.pth', help='모델 weight 파일 경로')
    parser.add_argument('--output-dir', default='/ailab_mat2/dataset/drone/250312_sejong/cmnextp_inference_ep30_cropped', help='시각화 결과 저장 디렉토리')
    # parser.add_argument('--num-samples', type=int, default=7796, help='시각화 및 평가할 샘플 수')
    parser.add_argument('--num-samples', type=int, default=200, help='시각화 및 평가할 샘플 수')
    parser.add_argument('--score-threshold', type=float, default=0.4, help='신뢰도 임계값')
    parser.add_argument('--device', default='cuda:0', help='사용할 디바이스')
    parser.add_argument('--draw-on', nargs='+', default=['rgb'], help='박스를 그릴 모달리티 지정 (e.g., rgb depth event lidar). 기본값: rgb')
    parser.add_argument(
        '--eval-only', 
        action='store_true',  # 이 플래그가 있으면 True가 됨
        help='이미지 저장을 생략하고 mAP 평가만 수행합니다.'
    )
    args = parser.parse_args()
    
    run_mode = 'eval_only' if args.eval_only else 'visualize'
    wandb.init(project='DELIVER', name=f'{run_mode}_cropped_{Path(args.checkpoint).stem}', tags=['inference', run_mode, 'cmnext', 'cropped'], config=vars(args))
    
    visualizer = SejongMultimodalVisualizer(config_path=args.config, checkpoint_path=args.checkpoint, device=args.device)
    
    # ▼▼▼ eval_only 인자 전달 ▼▼▼
    visualizer.visualize_from_validation_set(
        output_dir=args.output_dir, 
        num_samples=args.num_samples, 
        score_threshold=args.score_threshold, 
        draw_on_modalities=args.draw_on,
        eval_only=args.eval_only
    )
    wandb.finish()

if __name__ == '__main__':
    main()