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
from typing import Tuple, Dict

# 사용자의 커스텀 모듈(CMNextBackbone, SejongDetectionDataset 등)을 레지스트리에 등록
from mcdet import *

class SejongMultimodalVisualizer:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.cfg = mmengine.Config.fromfile(config_path)
        init_default_scope(self.cfg.get('default_scope', 'mmdet'))

        # 모델 초기화
        self.model = init_detector(config_path, checkpoint_path, device=device)
        
        # 클래스 및 색상 정보는 config 파일의 정확한 경로에서 가져오기
        self.classes = self.cfg.val_dataloader.dataset.metainfo.classes
        self.colors = self.cfg.val_dataloader.dataset.metainfo.palette

    def load_multimodal_images(self, rgb_img_path: str) -> Dict[str, np.ndarray]:
        """
        하나의 원본 RGB 이미지 경로를 기반으로 모든 모달리티 이미지를 시각화용으로 로드합니다.
        """
        p = Path(rgb_img_path)
        img_id = p.stem
        img_dir = p.parent

        # 참고: config에 따라 경로 규칙이 다를 수 있습니다.
        # SejongDetectionDataset의 경로 규칙에 맞춰 수정이 필요할 수 있습니다.
        modality_paths = {
            'rgb': str(p),
            'depth': str(p).replace('img', 'depth').replace('_rgb_', '_depth_'),
            'event': str(p).replace('img', 'event').replace('_rgb_', '_event_'), # ir -> event로 간주
            'lidar': str(p).replace('img', 'lidar').replace('_rgb_', '_lidar_') # intensity -> lidar로 간주
        }
        
        images = {}
        for modality, path in modality_paths.items():
            print(f"Loading {modality} image from: {path}")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None: continue
                # if modality == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # elif len(img.shape) == 2: # 그레이스케일
                #     img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[modality] = img
        return images

    def create_multimodal_concat(self, images: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """ Multimodal 이미지들을 2x2 그리드로 연결합니다. """
        if not images: raise ValueError("사용 가능한 이미지가 없습니다.")
        first_img = next(iter(images.values()))
        h, w = first_img.shape[:2]
        grid_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        modality_positions = {'rgb': (0, 0), 'depth': (0, 1), 'event': (1, 0), 'lidar': (1, 1)}
        
        for modality, (row, col) in modality_positions.items():
            if modality in images:
                img = images[modality]
                if img.shape[:2] != (h, w): img = cv2.resize(img, (w, h))
                if img.shape[-1] ==4: img = img[:, :, :3]  # 알파 채널 제거
                if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                grid_img[row*h:(row+1)*h, col*w:(col+1)*w] = img
        return grid_img, (h, w)
    
    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                   img_shape: Tuple[int, int], text_prefix: str,
                   box_format: str = 'xywh',  # 🔥 포맷을 지정하는 인자 추가!
                   base_color: Tuple[int, int, int] = None) -> np.ndarray:
        """ GT 또는 Prediction 바운딩 박스를 이미지에 그리는 범용 함수 """
        h, w = img_shape
        modality_offsets = {'rgb': (0, 0), 'depth': (0, w), 'event': (h, 0), 'lidar': (h, w)}
        
        for bbox, label in zip(boxes, labels):
            # ==================== 👇 여기가 수정된 부분입니다 👇 ====================
            # box_format에 따라 좌표를 올바르게 계산
            if box_format == 'xywh':
                x, y, box_w, box_h = bbox.astype(int)
                x1, y1, x2, y2 = x, y, x + box_w, y + box_h
            elif box_format == 'xyxy':
                x1, y1, x2, y2 = bbox.astype(int)
            else:
                raise ValueError(f"Unknown box_format: {box_format}")
            # =================================================================

            class_name = self.classes[label -1]
            color = base_color if base_color else self.colors[label % len(self.colors)]
            label_text = f'{text_prefix}: {class_name}'
            
            for offset_y, offset_x in modality_offsets.values():
                x1_s, y1_s, x2_s, y2_s = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
                cv2.rectangle(image, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1_s, y1_s - text_h - 5), (x1_s + text_w, y1_s), color, -1)
                cv2.putText(image, label_text, (x1_s, y1_s - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)   
        return image


    def add_modality_labels(self, concat_img: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """ 각 모달리티 영역에 라벨 추가 """
        h, w = img_shape
        modality_labels = {'RGB': (20, 30), 'Depth': (w + 20, 30), 'Event': (20, h + 30), 'LiDAR': (w + 20, h + 30)}
        for label, (x, y) in modality_labels.items():
            cv2.putText(concat_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return concat_img

    # @torch.no_grad()
    # def visualize_from_validation_set(self, output_dir: str, num_samples: int = 10, score_threshold: float = 0.3):
    #     """
    #     **Config 파일의 val_dataloader에서 얻은 텐서로 직접 추론하고 GT와 함께 시각화**
    #     """
    #     os.makedirs(output_dir, exist_ok=True)
    #     val_dataset_cfg = self.cfg.val_dataloader.dataset
    #     dataset = DATASETS.build(val_dataset_cfg)

    #     print(f"'{val_dataset_cfg.type}' 데이터셋에서 {len(dataset)}개의 샘플을 찾았습니다.")
    #     print(f"최대 {num_samples}개의 샘플에 대해 시각화를 진행합니다...")

    #     self.model.eval()
    #     for i, data in enumerate(dataset):
    #         if i >= num_samples: break

    #         # 1. 모델 추론: 데이터로더의 출력을 직접 모델에 전달
    #         # `inference_detector` 대신 model.test_step()을 모방하여 사용
            
    #         batched_data = {
    #             'inputs': [[item] for item in data['inputs']],
    #             'data_samples': [data['data_samples']]
    #         }
            
    #         processed_data = self.model.data_preprocessor(batched_data, training=False )
    #         predictions = self.model.forward(**processed_data, mode='predict')
            
    #         # 2. 결과 및 정보 추출
    #         pred_sample = predictions[0] # 배치 크기가 1이므로 첫 번째 결과 사용
    #         data_sample = data['data_samples']
            
    #         # 시각화를 위한 원본 RGB 이미지 경로 (리스트의 첫 번째 항목)
    #         rgb_img_path = data_sample.img_path[0]
    #         img_id = Path(rgb_img_path).stem
    #         output_path = os.path.join(output_dir, f'{img_id}_GT_and_Pred.jpg')
            
    #         print(f"[{i+1}/{num_samples}] 처리 중: {img_id}")

    #         # 3. 시각화용 원본 이미지들 로드 및 2x2 그리드 생성
    #         images = self.load_multimodal_images(rgb_img_path)
    #         if not images: continue
    #         concat_img, img_shape = self.create_multimodal_concat(images)

    #         # 4. 신뢰도 필터링된 예측 결과 추출
    #         pred_instances = pred_sample.pred_instances[pred_sample.pred_instances.scores > score_threshold]
    #         pred_boxes = pred_instances.bboxes.cpu().numpy()
    #         pred_labels = pred_instances.labels.cpu().numpy()

    #         # 5. Ground Truth 정보 추출
    #         gt_instances = data_sample.gt_instances
    #         gt_boxes = gt_instances.bboxes.cpu().numpy()
    #         gt_labels = gt_instances.labels.cpu().numpy()
            
    #         # 6. 박스 그리기 및 저장
    #         result_img = self.draw_boxes(concat_img, gt_boxes, gt_labels, img_shape, text_prefix="GT", base_color=(255, 255, 255))
    #         result_img = self.draw_boxes(result_img, pred_boxes, pred_labels, img_shape, text_prefix="Pred")
    #         result_img = self.add_modality_labels(result_img, img_shape)
    #         cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    #     print(f"\n시각화 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")

# multimodal_visualization.py 파일 내
    @torch.no_grad()
    def visualize_from_validation_set(self, output_dir: str, num_samples: int = 10, score_threshold: float = 0.3):
        os.makedirs(output_dir, exist_ok=True)
        val_dataset_cfg = self.cfg.val_dataloader.dataset
        dataset = DATASETS.build(val_dataset_cfg)

        print(f"'{val_dataset_cfg.type}' 데이터셋에서 {len(dataset)}개의 샘플을 찾았습니다.")
        print(f"최대 {num_samples}개의 샘플에 대해 시각화를 진행합니다...")

        self.model.eval()
        for i, data in enumerate(dataset):
            if i >= num_samples:
                break
            # ==================== 👇 여기가 수정된 부분입니다 (핵심) 👇 ====================
            
            # 1. 원본 GT 정보를 사용하기 위해, 처리 전의 data_sample을 저장합니다.
            original_data_sample = data['data_samples']

            # 2. 단일 샘플 데이터를 "크기 1의 배치" 형태로 변환합니다.
            batched_data = {
                'inputs': [[item] for item in data['inputs']],
                'data_samples': [original_data_sample] # 처리 전 원본 data_sample 전달
            }

            # 3. Data Preprocessor를 통해 '배치화된' 데이터 처리
            processed_data = self.model.data_preprocessor(batched_data, training=False)
            
            # 4. 처리된 데이터를 모델의 forward에 전달
            predictions = self.model.forward(**processed_data, mode='predict')

            # 5. 결과 및 정보 추출
            pred_sample = predictions[0]  # 배치 크기가 1이므로 항상 첫 번째 결과 사용
            
            # 처리 후의 data_sample에서 scale_factor를 가져옵니다.
            processed_data_sample = processed_data['data_samples'][0]
            scale_factor = processed_data_sample.metainfo['scale_factor']
            
            # 6. 신뢰도 필터링된 예측 결과 추출
            pred_instances = pred_sample.pred_instances[pred_sample.pred_instances.scores > score_threshold]
            pred_boxes = pred_instances.bboxes.cpu().numpy()
            pred_labels = pred_instances.labels.cpu().numpy()

            # 7. (🔥 중요) 예측된 Bbox를 원본 이미지 크기로 다시 스케일링합니다.
            # if pred_boxes.shape[0] > 0:
            #     # scale_factor (w_scale, h_scale)를 [w_s, h_s, w_s, h_s] 형태로 만들어 한번에 나눗셈
            #     rescale_factor = np.tile(scale_factor, 2)
            #     pred_boxes[:, :4] /= rescale_factor

            # gt_instances = original_data_sample.gt_instances 
            # gt_boxes = gt_instances.bboxes.cpu().numpy()
            # gt_labels = gt_instances.labels.cpu().numpy()
            
            # ============================================================================

            # 시각화를 위한 원본 RGB 이미지 경로 (리스트의 첫 번째 항목)
            rgb_img_path = original_data_sample.img_path[0]
            img_id = Path(rgb_img_path).stem
            output_path = os.path.join(output_dir, f'{img_id}_GT_and_Pred.jpg')
            
            print(f"[{i+1}/{num_samples}] 처리 중: {img_id}")

            # 시각화용 원본 이미지들 로드 및 2x2 그리드 생성
            images = self.load_multimodal_images(rgb_img_path)
            if not images: continue
            concat_img, img_shape = self.create_multimodal_concat(images)

            # 박스 그리기 및 저장
            # result_img = self.draw_boxes(concat_img, gt_boxes, gt_labels, img_shape, text_prefix="GT", base_color=(255, 255, 255))
            # result_img = self.draw_boxes(concat_img, pred_boxes, pred_labels, img_shape, text_prefix="Pred")
            # result_img = self.draw_boxes(concat_img, gt_boxes, gt_labels, img_shape, 
                                # text_prefix="GT", base_color=(255, 255, 255), box_format='xywh')
            result_img = self.draw_boxes(concat_img, pred_boxes, pred_labels, img_shape, 
                                         text_prefix="Pred", box_format='xyxy')
            result_img = self.add_modality_labels(result_img, img_shape)
            cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

        print(f"\n시각화 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")

def main():
    # ... (main 함수는 변경 없음) ...
    parser = argparse.ArgumentParser(description='Sejong Multimodal Detection Visualization based on Validation Set')
    parser.add_argument('--config', default='/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/deliver_cmnext_b2_faster_rcnn_2x_cosinelr0.01_ep50/hinton-deliver_cmnext_rcnn_lr0.01_ep50.py', help='모델 config 파일 경로')
    parser.add_argument('--checkpoint', default='/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/deliver_cmnext_b2_faster_rcnn_2x_cosinelr0.01_ep50/best_coco_bbox_mAP_epoch_45.pth', help='모델 weight 파일 경로')
    parser.add_argument('--output-dir', default='/ailab_mat2/personal/jemo_maeng/dset/Drone/CMNeXT/inference/deliver_cmnext_b2_faster_rcnn_2x_cosinelr0.01_ep50', help='시각화 결과 저장 디렉토리')
    parser.add_argument('--num-samples', type=int, default=500, help='시각화할 샘플 수')
    parser.add_argument('--score-threshold', type=float, default=0.5, help='신뢰도 임계값')
    parser.add_argument('--device', default='cuda:0', help='사용할 디바이스')
    args = parser.parse_args()
    
    visualizer = SejongMultimodalVisualizer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    visualizer.visualize_from_validation_set(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        score_threshold=args.score_threshold
    )

if __name__ == '__main__':
    main()