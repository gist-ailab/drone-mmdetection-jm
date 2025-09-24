#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import mmengine
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from typing import Tuple, Dict, List
import wandb

# 사용자의 커스텀 모듈(CMNextBackbone, SejongDetectionDataset 등)을 레지스트리에 등록
from mcdet import *

class SejongMultimodalVisualizer:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.cfg = mmengine.Config.fromfile(config_path)
        init_default_scope(self.cfg.get('default_scope', 'mmdet'))

        self.model = init_detector(config_path, checkpoint_path, device=device, cfg_options={'strict': False})
        
        # 클래스 및 색상 정보 로드
        self.classes = self.cfg.val_dataloader.dataset.metainfo.classes
        self.colors = self.cfg.val_dataloader.dataset.metainfo.palette

    def load_multimodal_images(self, rgb_img_path: str) -> Dict[str, np.ndarray]:
        """하나의 원본 RGB 이미지 경로를 기반으로 모든 모달리티 이미지를 시각화용으로 로드합니다."""
        p = Path(rgb_img_path)
        img_id = p.stem
        
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

                if modality == 'rgb':
                    pass
                elif len(img.shape) == 2: # 그레이스케일
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                images[modality] = img
        return images

    def create_multimodal_concat(self, images: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """ Multimodal 이미지들을 2x2 그리드로 연결합니다. """
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
                   box_format: str = 'xywh',
                   base_color: Tuple[int, int, int] = None,
                   draw_on: List[str] = None) -> np.ndarray:
        h, w = img_shape
        modality_offsets = {'rgb': (0, 0), 'depth': (0, w), 'event': (h, 0), 'lidar': (h, w)}
        target_modalities = draw_on if draw_on else modality_offsets.keys()
        
        for bbox, label in zip(boxes, labels):
            if box_format == 'xywh':
                x, y, box_w, box_h = bbox.astype(int)
                x1, y1, x2, y2 = x, y, x + box_w, y + box_h
            elif box_format == 'xyxy':
                x1, y1, x2, y2 = bbox.astype(int)
            else:
                raise ValueError(f"Unknown box_format: {box_format}")

            class_name = self.classes[label]
            color = base_color if base_color else self.colors[label % len(self.colors)]
            label_text = f'{text_prefix}: {class_name}'
            
            for modality in target_modalities:
                if modality not in modality_offsets:
                    continue
                
                offset_y, offset_x = modality_offsets[modality]
                x1_s, y1_s, x2_s, y2_s = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
                
                cv2.rectangle(image, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1_s, y1_s - text_h - 5), (x1_s + text_w, y1_s), color, -1)
                cv2.putText(image, label_text, (x1_s, y1_s - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return image

    def add_modality_labels(self, concat_img: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        h, w = img_shape
        modality_labels = {'RGB': (20, 30), 'Depth': (w + 20, 30), 'IR': (20, h + 30), 'LiDAR': (w + 20, h + 30)}
        for label, (x, y) in modality_labels.items():
            cv2.putText(concat_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return concat_img

    @torch.no_grad()
    def visualize_from_validation_set(self, output_dir: str, num_samples: int = 10, score_threshold: float = 0.3,
                                      draw_on_modalities: List[str] = None):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'w_gt'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'wo_gt'), exist_ok=True)
        
        val_dataset_cfg = self.cfg.val_dataloader.dataset
        dataset = DATASETS.build(val_dataset_cfg)

        print(f"'{val_dataset_cfg.type}' 데이터셋에서 {len(dataset)}개의 샘플을 찾았습니다.")
        print(f"최대 {num_samples}개의 샘플에 대해 시각화 및 평가를 진행합니다...")
        print(f"Bounding box는 다음 모달리티에 그려집니다: {draw_on_modalities}")

        self.model.eval()
        
        ## ▼▼▼ mAP 계산 추가 1 ▼▼▼
        # 모든 예측 결과를 저장할 리스트를 초기화합니다.
        all_predictions = []
        ## ▲▲▲ mAP 계산 추가 1 ▲▲▲

        for i, data in enumerate(dataset):
            if i >= num_samples:
                break
            
            original_data_sample = data['data_samples']
            batched_data = { 'inputs': [[item] for item in data['inputs']], 'data_samples': [original_data_sample] }
            processed_data = self.model.data_preprocessor(batched_data, training=False)
            predictions = self.model.forward(**processed_data, mode='predict')

            ## ▼▼▼ mAP 계산 추가 2 ▼▼▼
            # 현재 배치의 예측 결과를 전체 예측 리스트에 추가합니다.
            all_predictions.extend(predictions)
            ## ▲▲▲ mAP 계산 추가 2 ▲▲▲

            pred_sample = predictions[0]
            processed_data_sample = processed_data['data_samples'][0]
            scale_factor = processed_data_sample.metainfo['scale_factor']
            
            pred_instances = pred_sample.pred_instances[pred_sample.pred_instances.scores > score_threshold]
            pred_boxes = pred_instances.bboxes.cpu().numpy()
            pred_labels = pred_instances.labels.cpu().numpy()

            if pred_boxes.shape[0] > 0:
                rescale_factor = np.tile(scale_factor, 2)
                pred_boxes[:, :4] /= rescale_factor

            gt_instances = original_data_sample.gt_instances 
            gt_boxes = gt_instances.bboxes.cpu().numpy()
            gt_labels = gt_instances.labels.cpu().numpy()
            
            rgb_img_path = original_data_sample.img_path[0]
            img_id = Path(rgb_img_path).stem
            output_path_wgt = os.path.join(output_dir, 'w_gt', f'{img_id}.jpg')
            output_path_wogt = os.path.join(output_dir, 'wo_gt', f'{img_id}.jpg')
            
            print(f"[{i+1}/{num_samples}] 처리 중: {img_id}")

            images = self.load_multimodal_images(rgb_img_path)
            if not images: continue
            concat_img, img_shape = self.create_multimodal_concat(images)

            result_img = self.draw_boxes(concat_img.copy(), pred_boxes, pred_labels, img_shape, 
                                         text_prefix="Pred", box_format='xyxy', 
                                         draw_on=draw_on_modalities)
            result_img = self.add_modality_labels(result_img, img_shape)
            cv2.imwrite(output_path_wogt, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            
            result_img_w_gt = self.draw_boxes(result_img, gt_boxes, gt_labels, img_shape, 
                                              text_prefix="GT", base_color=(255, 255, 255), box_format='xywh', 
                                              draw_on=draw_on_modalities)
            cv2.imwrite(output_path_wgt, cv2.cvtColor(result_img_w_gt, cv2.COLOR_RGB2BGR))
        
        print(f"\n시각화 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")

        ## ▼▼▼ mAP 계산 추가 3 ▼▼▼
        if all_predictions:
            print(f"\n{len(all_predictions)}개의 샘플에 대한 평가(mAP)를 시작합니다...")
            
            # config에 정의된 evaluator를 사용하여 평가를 수행합니다.
            # metric='bbox'는 바운딩 박스 detection 성능을 평가하겠다는 의미입니다.
            eval_metrics = dataset.evaluate(all_predictions, metric='bbox')

            print("\n---*--- 평가 결과 ---*---")
            # 주요 지표들을 보기 쉽게 출력합니다.
            for key, value in eval_metrics.items():
                #wandb 로깅을 위해 소수점 4자리까지 float으로 변환
                if isinstance(value, np.float32):
                    eval_metrics[key] = round(float(value), 4)
                print(f"{key:<25}: {eval_metrics[key]}")
            print("---*-----------------*---")

            # wandb에 평가 지표 로깅
            if wandb.run:
                # wandb는 접두사를 사용하여 그룹화하는 것을 지원합니다.
                wandb_metrics = {f'eval/{k}': v for k, v in eval_metrics.items()}
                wandb.log(wandb_metrics)
                print("\n평가 지표를 wandb에 성공적으로 로깅했습니다. (eval/ 접두사 사용)")
        else:
            print("\n평가할 예측 결과가 없습니다.")
        ## ▲▲▲ mAP 계산 추가 3 ▲▲▲


def main():
    parser = argparse.ArgumentParser(description='Sejong Multimodal Detection Visualization and Evaluation')
    parser.add_argument('--config', default='/SSDb/jemo_maeng/src/Project/Drone/detection/drone-mmdetection-jm/work_dirs/sejong2504_cmnextp_b2_rcnn_multiscale_v2/yeon-sejong2504_cmnextp_rcnn_lr0.01_ep50_v2.py', help='모델 config 파일 경로')
    parser.add_argument('--checkpoint', default='/SSDb/jemo_maeng/src/Project/Drone/detection/drone-mmdetection-jm/work_dirs/sejong2504_cmnextp_b2_rcnn_multiscale_v2/best_coco_bbox_mAP_epoch_30.pth', help='모델 weight 파일 경로')
    parser.add_argument('--output-dir', default='/ailab_mat2/dataset/drone/250312_sejong/cmnextp_inference_ep30', help='시각화 결과 저장 디렉토리')
    parser.add_argument('--num-samples', type=int, default=7796, help='시각화 및 평가할 샘플 수')
    parser.add_argument('--score-threshold', type=float, default=0.4, help='신뢰도 임계값')
    parser.add_argument('--device', default='cuda:0', help='사용할 디바이스')
    parser.add_argument('--draw-on', nargs='+', default=['rgb'], 
                        help='박스를 그릴 모달리티 지정 (e.g., rgb depth event lidar). 기본값: rgb')
                        
    args = parser.parse_args()
    
    wandb.init(
        project='DELIVER',
        name=f'eval_{Path(args.checkpoint).stem}', # 이름에 eval 명시
        tags=['inference', 'visualization', 'evaluation', 'cmnext'], # evaluation 태그 추가
        config=vars(args)
    )
    
    visualizer = SejongMultimodalVisualizer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    visualizer.visualize_from_validation_set(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        score_threshold=args.score_threshold,
        draw_on_modalities=args.draw_on
    )
    wandb.finish() # 평가 후 wandb 세션 종료

if __name__ == '__main__':
    main()