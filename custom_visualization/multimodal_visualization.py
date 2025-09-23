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
                    # BGR to RGB 변환은 최종 저장 시점에 한번만 수행하는 것이 효율적입니다.
                    # 여기서는 BGR로 로드하고, 시각화 함수에서 RGB로 변환하여 처리합니다.
                    pass
                elif len(img.shape) == 2: # 그레이스케일
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                images[modality] = img
        return images

    def create_multimodal_concat(self, images: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """ Multimodal 이미지들을 2x2 그리드로 연결합니다. """
        if not images: raise ValueError("사용 가능한 이미지가 없습니다.")
        
        # RGB 이미지를 기준으로 크기 통일
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
                
                # RGB 이미지는 BGR -> RGB 변환 후 배치
                if modality == 'rgb':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                grid_img[row*h:(row+1)*h, col*w:(col+1)*w] = img
        return grid_img, (h, w)
    
    ## ▼▼▼ 개선된 부분 1 ▼▼▼
    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                   img_shape: Tuple[int, int], text_prefix: str,
                   box_format: str = 'xywh',
                   base_color: Tuple[int, int, int] = None,
                   draw_on: List[str] = None) -> np.ndarray: # 'draw_on' 인자 추가
        """ 
        GT 또는 Prediction 바운딩 박스를 이미지에 그리는 범용 함수.
        - 요청 1: base_color가 None이면(Prediction의 경우), self.colors에서 클래스별 색상을 가져옵니다.
        - 요청 2: draw_on 리스트에 명시된 모달리티에만 박스를 그립니다.
        """
        h, w = img_shape
        modality_offsets = {'rgb': (0, 0), 'depth': (0, w), 'event': (h, 0), 'lidar': (h, w)}

        # draw_on이 None이거나 비어있으면 모든 모달리티에 그리도록 처리
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
            
            # 🚀 [개선 1] base_color가 지정되지 않으면(예: Prediction), 클래스 팔레트에서 색상을 가져옵니다.
            color = base_color if base_color else self.colors[label % len(self.colors)]
            label_text = f'{text_prefix}: {class_name}'
            
            # 🚀 [개선 2] 지정된 모달리티에만 박스를 그립니다.
            for modality in target_modalities:
                if modality not in modality_offsets:
                    continue # 유효하지 않은 모달리티 이름은 건너뜁니다.
                
                offset_y, offset_x = modality_offsets[modality]
                x1_s, y1_s, x2_s, y2_s = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
                
                cv2.rectangle(image, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1_s, y1_s - text_h - 5), (x1_s + text_w, y1_s), color, -1)
                cv2.putText(image, label_text, (x1_s, y1_s - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
        return image
    ## ▲▲▲ 개선된 부분 1 ▲▲▲

    def add_modality_labels(self, concat_img: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """ 각 모달리티 영역에 라벨 추가 """
        h, w = img_shape
        modality_labels = {'RGB': (20, 30), 'Depth': (w + 20, 30), 'IR': (20, h + 30), 'LiDAR': (w + 20, h + 30)}
        for label, (x, y) in modality_labels.items():
            cv2.putText(concat_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return concat_img

    @torch.no_grad()
    def visualize_from_validation_set(self, output_dir: str, num_samples: int = 10, score_threshold: float = 0.3,
                                      draw_on_modalities: List[str] = None): # 'draw_on_modalities' 인자 추가
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'w_gt'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'wo_gt'), exist_ok=True)
        
        val_dataset_cfg = self.cfg.val_dataloader.dataset
        dataset = DATASETS.build(val_dataset_cfg)

        print(f"'{val_dataset_cfg.type}' 데이터셋에서 {len(dataset)}개의 샘플을 찾았습니다.")
        print(f"최대 {num_samples}개의 샘플에 대해 시각화를 진행합니다...")
        print(f"Bounding box는 다음 모달리티에 그려집니다: {draw_on_modalities}")

        self.model.eval()
        for i, data in enumerate(dataset):
            if i >= num_samples:
                break
            
            original_data_sample = data['data_samples']

            batched_data = {
                'inputs': [[item] for item in data['inputs']],
                'data_samples': [original_data_sample]
            }

            processed_data = self.model.data_preprocessor(batched_data, training=False)
            predictions = self.model.forward(**processed_data, mode='predict')

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

            ## ▼▼▼ 개선된 부분 2 ▼▼▼
            # draw_boxes 함수에 draw_on_modalities 인자를 전달합니다.
            # Prediction 박스는 base_color를 지정하지 않아 클래스별 색상이 적용됩니다.
            result_img = self.draw_boxes(concat_img.copy(), pred_boxes, pred_labels, img_shape, 
                                         text_prefix="Pred", box_format='xyxy', 
                                         draw_on=draw_on_modalities)
            result_img = self.add_modality_labels(result_img, img_shape)
            cv2.imwrite(output_path_wogt, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            
            # GT 박스는 흰색(base_color)으로, 동일한 모달리티에 그려집니다.
            result_img_w_gt = self.draw_boxes(result_img, gt_boxes, gt_labels, img_shape, 
                                              text_prefix="GT", base_color=(255, 255, 255), box_format='xywh', 
                                              draw_on=draw_on_modalities)
            cv2.imwrite(output_path_wgt, cv2.cvtColor(result_img_w_gt, cv2.COLOR_RGB2BGR))
            ## ▲▲▲ 개선된 부분 2 ▲▲▲

        print(f"\n시각화 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='Sejong Multimodal Detection Visualization based on Validation Set')
    parser.add_argument('--config', default='/SSDb/jemo_maeng/src/Project/Drone/detection/drone-mmdetection-jm/work_dirs/sejong2504_cmnextp_b2_rcnn_multiscale_v2/yeon-sejong2504_cmnextp_rcnn_lr0.01_ep50_v2.py', help='모델 config 파일 경로')
    parser.add_argument('--checkpoint', default='/SSDb/jemo_maeng/src/Project/Drone/detection/drone-mmdetection-jm/work_dirs/sejong2504_cmnextp_b2_rcnn_multiscale_v2/best_coco_bbox_mAP_epoch_30.pth', help='모델 weight 파일 경로')
    parser.add_argument('--output-dir', default='/ailab_mat2/dataset/drone/250312_sejong/cmnextp_inference_ep30', help='시각화 결과 저장 디렉토리')
    parser.add_argument('--num-samples', type=int, default=7796, help='시각화할 샘플 수')
    parser.add_argument('--score-threshold', type=float, default=0.4, help='신뢰도 임계값')
    parser.add_argument('--device', default='cuda:0', help='사용할 디바이스')
    
    ## ▼▼▼ 개선된 부분 3 ▼▼▼
    # nargs='+'는 하나 이상의 인자를 리스트로 받습니다.
    # default=['rgb']로 기본값은 RGB 이미지만 설정합니다.
    parser.add_argument('--draw-on', nargs='+', default=['rgb'], 
                        help='박스를 그릴 모달리티 지정 (e.g., rgb depth event lidar). 기본값: rgb')
    ## ▲▲▲ 개선된 부분 3 ▲▲▲
                        
    args = parser.parse_args()
    
    wandb.init(
        project='DELIVER',
        name=f'visualization_{Path(args.checkpoint).stem}',
        tags=['inference', 'visualization', 'cmnext'],
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
        draw_on_modalities=args.draw_on # 파싱된 인자를 전달합니다.
    )

if __name__ == '__main__':
    main()