#!/usr/bin/env python3
"""
CMNeXtPSP 모델 분석을 위한 시각화 및 평가 스크립트
- 데이터 시각화 및 저장
- mAP 수치 계산
- FRM 모듈 선택 분석
"""

import os
import cv2
import torch
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import mmengine
from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from mmdet.apis import inference_detector
from mmdet.evaluation import CocoMetric
import wandb

# 사용자의 커스텀 모듈 등록
from mcdet import *

class CMNeXtPSPAnalyzer:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.cfg = mmengine.Config.fromfile(config_path)
        mmengine.registry.init_default_scope(self.cfg.get('default_scope', 'mmdet'))
        self.model = init_detector(config_path, checkpoint_path, device=device, cfg_options={'strict': False})
        self.classes = self.cfg.val_dataloader.dataset.metainfo.classes
        self.colors = self.cfg.val_dataloader.dataset.metainfo.palette
        
        # FRM/FFM 모듈 분석을 위한 훅 등록
        self.frm_analysis_data = {}
        self.ffm_analysis_data = {}
        self.register_analysis_hooks()
    
    def register_analysis_hooks(self):
        """FRM/FFM 모듈 분석을 위한 훅 등록"""
        def frm_hook(module, input, output):
            if hasattr(module, '__name__') and 'FRM' in str(module.__class__):
                stage_idx = getattr(module, '_stage_idx', 0)
                if stage_idx not in self.frm_analysis_data:
                    self.frm_analysis_data[stage_idx] = []
                
                # 입력과 출력 분석
                if isinstance(input, (list, tuple)) and len(input) >= 2:
                    x1, x2 = input[0], input[1]
                    if isinstance(output, (list, tuple)) and len(output) >= 2:
                        out_x1, out_x2 = output[0], output[1]
                        
                        # 채널별 가중치 분석
                        channel_weights = module.channel_weights(x1, x2)
                        spatial_weights = module.spatial_weights(x1, x2)
                        
                        analysis = {
                            'stage': stage_idx,
                            'input_shapes': [x1.shape, x2.shape],
                            'output_shapes': [out_x1.shape, out_x2.shape],
                            'channel_weights_mean': [w.mean().item() for w in channel_weights],
                            'spatial_weights_mean': [w.mean().item() for w in spatial_weights],
                            'lambda_c': module.lambda_c,
                            'lambda_s': module.lambda_s
                        }
                        self.frm_analysis_data[stage_idx].append(analysis)
        
        def ffm_hook(module, input, output):
            if hasattr(module, '__class__') and 'FeatureFusionModule' in str(module.__class__):
                stage_idx = getattr(module, '_stage_idx', 0)
                if stage_idx not in self.ffm_analysis_data:
                    self.ffm_analysis_data[stage_idx] = []
                
                if isinstance(input, (list, tuple)) and len(input) >= 2:
                    x1, x2 = input[0], input[1]
                    analysis = {
                        'stage': stage_idx,
                        'input_shapes': [x1.shape, x2.shape],
                        'output_shape': output.shape,
                        'feature_fusion_ratio': (x1.mean() / (x2.mean() + 1e-8)).item()
                    }
                    self.ffm_analysis_data[stage_idx].append(analysis)
        
        # 모델의 FRM과 FFM 모듈에 훅 등록
        for name, module in self.model.named_modules():
            if 'FRM' in str(module.__class__):
                # 스테이지 인덱스 추출 (안전하게)
                stage_idx = 0
                try:
                    parts = name.split('.')
                    for part in parts:
                        if part.isdigit():
                            stage_idx = int(part)
                            break
                except (ValueError, IndexError):
                    stage_idx = 0
                module._stage_idx = stage_idx
                module.register_forward_hook(frm_hook)
            elif 'FeatureFusionModule' in str(module.__class__):
                # 스테이지 인덱스 추출 (안전하게)
                stage_idx = 0
                try:
                    parts = name.split('.')
                    for part in parts:
                        if part.isdigit():
                            stage_idx = int(part)
                            break
                except (ValueError, IndexError):
                    stage_idx = 0
                module._stage_idx = stage_idx
                module.register_forward_hook(ffm_hook)
    
    def load_multimodal_images(self, rgb_img_path: str) -> Dict[str, np.ndarray]:
        """멀티모달 이미지 로드"""
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
                if img is None: 
                    continue
                if len(img.shape) == 2:
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                images[modality] = img
        return images
    
    def create_multimodal_concat(self, images: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """멀티모달 이미지를 2x2 그리드로 결합"""
        if not images: 
            raise ValueError("사용 가능한 이미지가 없습니다.")
        ref_img = images.get('rgb')
        if ref_img is None: 
            ref_img = next(iter(images.values()))
        h, w = ref_img.shape[:2]
        grid_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        modality_positions = {'rgb': (0, 0), 'depth': (0, 1), 'event': (1, 0), 'lidar': (1, 1)}
        
        for modality, (row, col) in modality_positions.items():
            if modality in images:
                img = images[modality]
                if img.shape[:2] != (h, w): 
                    img = cv2.resize(img, (w, h))
                if len(img.shape) == 2: 
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if modality == 'rgb':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                grid_img[row*h:(row+1)*h, col*w:(col+1)*w] = img
        return grid_img, (h, w)
    
    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                   img_shape: Tuple[int, int], text_prefix: str,
                   box_format: str = 'xyxy',
                   draw_on: List[str] = None) -> np.ndarray:
        """바운딩 박스 그리기"""
        h, w = img_shape
        modality_offsets = {'rgb': (0, 0), 'depth': (0, w), 'event': (h, 0), 'lidar': (h, w)}
        target_modalities = draw_on if draw_on else modality_offsets.keys()
        
        for bbox, label in zip(boxes, labels):
            if box_format == 'xywh':
                x, y, box_w, box_h = bbox.astype(int)
                x1, y1, x2, y2 = x, y, x + box_w, y + box_h
            else: 
                x1, y1, x2, y2 = bbox.astype(int)
            
            class_name = self.classes[label]
            color = self.colors[label % len(self.colors)]
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
    
    @torch.no_grad()
    def run_analysis(self, output_dir: str, num_samples: int = 10, score_threshold: float = 0.5):
        """전체 분석 실행"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'analysis'), exist_ok=True)
        
        # 데이터셋 로드
        val_dataset_cfg = self.cfg.val_dataloader.dataset
        dataset = DATASETS.build(val_dataset_cfg)
        dataset.full_init()
        
        total_samples = len(dataset) if num_samples < 0 else min(num_samples, len(dataset))
        print(f"총 {total_samples}개의 샘플에 대해 분석을 진행합니다...")
        
        self.model.eval()
        coco_results = []
        
        for i, data in enumerate(dataset):
            if i >= total_samples: 
                break
            
            print(f"[{i+1}/{total_samples}] 처리 중...")
            
            # 추론 실행
            original_data_sample = data['data_samples']
            image_id = original_data_sample.img_id
            
            batched_data = { 
                'inputs': [[item] for item in data['inputs']], 
                'data_samples': [original_data_sample] 
            }
            processed_data = self.model.data_preprocessor(batched_data, training=False)
            predictions = self.model.forward(**processed_data, mode='predict')
            pred_sample = predictions[0]
            
            # 예측 결과 처리
            scale_factor = processed_data['data_samples'][0].metainfo['scale_factor']
            pred_instances = pred_sample.pred_instances[pred_sample.pred_instances.scores > score_threshold]
            
            pred_boxes_xyxy = pred_instances.bboxes.cpu().numpy() / np.tile(scale_factor, 2)
            pred_labels = pred_instances.labels.cpu().numpy()
            pred_scores = pred_instances.scores.cpu().numpy()
            
            # COCO 형식으로 결과 저장
            for box, label, score in zip(pred_boxes_xyxy, pred_labels, pred_scores):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                category_id = dataset.cat_ids[label]
                coco_results.append({
                    'image_id': image_id, 
                    'category_id': category_id,
                    'bbox': [float(coord) for coord in [x1, y1, w, h]],
                    'score': float(score)
                })
            
            # 시각화
            rgb_img_path = original_data_sample.img_path[0]
            img_id_str = Path(rgb_img_path).stem
            
            images = self.load_multimodal_images(rgb_img_path)
            if not images: 
                continue
            
            concat_img, img_shape = self.create_multimodal_concat(images)
            result_img = self.draw_boxes(concat_img.copy(), pred_boxes_xyxy, pred_labels, img_shape, 
                                       text_prefix="Pred", box_format='xyxy', draw_on=['rgb'])
            
            # 결과 저장
            cv2.imwrite(os.path.join(output_dir, 'visualizations', f'{img_id_str}.jpg'), 
                       cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
        # COCO 결과 저장
        with open(os.path.join(output_dir, 'coco_predictions.json'), 'w') as f:
            json.dump(coco_results, f, indent=4)
        
        print(f"시각화 완료: {output_dir}/visualizations/")
        print(f"예측 결과 저장: {output_dir}/coco_predictions.json")
        
        return coco_results
    
    def analyze_frm_selection(self, output_dir: str):
        """FRM 모듈 선택 분석"""
        print("FRM 모듈 선택 분석 중...")
        
        # FRM 분석 데이터 시각화
        if self.frm_analysis_data:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('FRM Module Analysis', fontsize=16)
            
            for stage_idx, data_list in self.frm_analysis_data.items():
                if not data_list:
                    continue
                
                # 채널 가중치 분석
                channel_weights_0 = [d['channel_weights_mean'][0] for d in data_list]
                channel_weights_1 = [d['channel_weights_mean'][1] for d in data_list]
                
                row, col = stage_idx // 2, stage_idx % 2
                if row < 2 and col < 2:
                    axes[row, col].plot(channel_weights_0, label='Channel Weight 0', alpha=0.7)
                    axes[row, col].plot(channel_weights_1, label='Channel Weight 1', alpha=0.7)
                    axes[row, col].set_title(f'Stage {stage_idx + 1} - Channel Weights')
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'analysis', 'frm_channel_weights.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # FRM 선택 통계 저장
            frm_stats = {}
            for stage_idx, data_list in self.frm_analysis_data.items():
                if data_list:
                    frm_stats[f'stage_{stage_idx}'] = {
                        'avg_channel_weight_0': np.mean([d['channel_weights_mean'][0] for d in data_list]),
                        'avg_channel_weight_1': np.mean([d['channel_weights_mean'][1] for d in data_list]),
                        'avg_spatial_weight_0': np.mean([d['spatial_weights_mean'][0] for d in data_list]),
                        'avg_spatial_weight_1': np.mean([d['spatial_weights_mean'][1] for d in data_list]),
                        'lambda_c': data_list[0]['lambda_c'],
                        'lambda_s': data_list[0]['lambda_s']
                    }
            
            with open(os.path.join(output_dir, 'analysis', 'frm_analysis.json'), 'w') as f:
                json.dump(frm_stats, f, indent=4)
            
            print(f"FRM 분석 결과 저장: {output_dir}/analysis/frm_analysis.json")
    
    def analyze_ffm_fusion(self, output_dir: str):
        """FFM 융합 분석"""
        print("FFM 융합 분석 중...")
        
        if self.ffm_analysis_data:
            # FFM 융합 비율 분석
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('FFM Fusion Analysis', fontsize=16)
            
            for stage_idx, data_list in self.ffm_analysis_data.items():
                if not data_list:
                    continue
                
                fusion_ratios = [d['feature_fusion_ratio'] for d in data_list]
                
                row, col = stage_idx // 2, stage_idx % 2
                if row < 2 and col < 2:
                    axes[row, col].hist(fusion_ratios, bins=20, alpha=0.7, edgecolor='black')
                    axes[row, col].set_title(f'Stage {stage_idx + 1} - Feature Fusion Ratio')
                    axes[row, col].set_xlabel('Fusion Ratio (x1_mean / x2_mean)')
                    axes[row, col].set_ylabel('Frequency')
                    axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'analysis', 'ffm_fusion_ratios.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # FFM 통계 저장
            ffm_stats = {}
            for stage_idx, data_list in self.ffm_analysis_data.items():
                if data_list:
                    fusion_ratios = [d['feature_fusion_ratio'] for d in data_list]
                    ffm_stats[f'stage_{stage_idx}'] = {
                        'avg_fusion_ratio': np.mean(fusion_ratios),
                        'std_fusion_ratio': np.std(fusion_ratios),
                        'min_fusion_ratio': np.min(fusion_ratios),
                        'max_fusion_ratio': np.max(fusion_ratios)
                    }
            
            with open(os.path.join(output_dir, 'analysis', 'ffm_analysis.json'), 'w') as f:
                json.dump(ffm_stats, f, indent=4)
            
            print(f"FFM 분석 결과 저장: {output_dir}/analysis/ffm_analysis.json")
    
    def evaluate_map(self, output_dir: str, coco_results: List[Dict]):
        """mAP 평가"""
        print("mAP 평가 중...")
        
        # COCO 평가 실행
        val_dataset_cfg = self.cfg.val_dataloader.dataset
        dataset = DATASETS.build(val_dataset_cfg)
        dataset.full_init()
        
        # COCO 형식으로 결과 저장
        results_file = os.path.join(output_dir, 'coco_predictions.json')
        
        # mmdet의 CocoMetric을 사용한 평가
        evaluator = CocoMetric(
            ann_file=val_dataset_cfg.ann_file,
            metric='bbox',
            format_only=False
        )
        
        # 평가 실행 (실제 구현에서는 더 복잡한 과정이 필요)
        print(f"COCO 형식 예측 결과가 {results_file}에 저장되었습니다.")
        print("별도의 평가 스크립트를 사용하여 mAP를 계산하세요.")
        
        return results_file

def main():
    parser = argparse.ArgumentParser(description='CMNeXtPSP Model Analysis')
    parser.add_argument('--config', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_heuristicalign_cmnextpsp_b2_rcnn_multiscale_v2_with_visualization/local-sejong2504_heuristicalign_cmnextpsp_rcnn_lr0.01_ep50_v2_with_visualization.py',
                       help='Model config file path')
    parser.add_argument('--checkpoint', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_heuristicalign_cmnextpsp_b2_rcnn_multiscale_v2_with_visualization/epoch_15.pth',
                       help='Model checkpoint file path')
    parser.add_argument('--output-dir', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of samples to analyze')
    parser.add_argument('--score-threshold', type=float, default=0.5, help='Score threshold for predictions')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    
    args = parser.parse_args()
    
    # Wandb 초기화
    wandb.init(
        project='CMNeXtPSP-Analysis',
        name=f'analysis_{Path(args.checkpoint).stem}',
        tags=['analysis', 'frm', 'ffm', 'visualization'],
        config=vars(args)
    )
    
    # 분석기 초기화
    analyzer = CMNeXtPSPAnalyzer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # 분석 실행
    print("=== CMNeXtPSP 모델 분석 시작 ===")
    
    # 1. 데이터 시각화 및 예측 저장
    print("1. 데이터 시각화 및 예측 저장...")
    coco_results = analyzer.run_analysis(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        score_threshold=args.score_threshold
    )
    
    # 2. FRM 모듈 선택 분석
    print("2. FRM 모듈 선택 분석...")
    analyzer.analyze_frm_selection(args.output_dir)
    
    # 3. FFM 융합 분석
    print("3. FFM 융합 분석...")
    analyzer.analyze_ffm_fusion(args.output_dir)
    
    # 4. mAP 평가
    print("4. mAP 평가...")
    results_file = analyzer.evaluate_map(args.output_dir, coco_results)
    
    print(f"\n=== 분석 완료 ===")
    print(f"결과 디렉토리: {args.output_dir}")
    print(f"시각화: {args.output_dir}/visualizations/")
    print(f"분석 결과: {args.output_dir}/analysis/")
    print(f"예측 결과: {results_file}")
    
    wandb.finish()

if __name__ == '__main__':
    main()
