#!/usr/bin/env python3
"""
종합적인 모델 평가 스크립트
- Inference 이미지 저장
- Ground Truth 레이블과 함께 시각화
- CMNeXt Visualization Hook 정보 저장
- mAP 평가
"""

import os
import json
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import mmengine
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import DATASETS, VISUALIZERS
from mmdet.evaluation import CocoMetric
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer
import pickle
from datetime import datetime
import shutil

# 사용자의 커스텀 모듈 등록
from mcdet import *


class ComprehensiveEvaluator:
    """종합적인 모델 평가를 위한 클래스"""
    
    def __init__(self, config_path: str, checkpoint_path: str, 
                 output_dir: str, device: str = 'cuda:0'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.device = device
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        self.inference_dir = os.path.join(output_dir, 'inference_images')
        self.gt_dir = os.path.join(output_dir, 'gt_visualizations')
        self.hook_dir = os.path.join(output_dir, 'hook_visualizations')
        self.results_dir = os.path.join(output_dir, 'evaluation_results')
        
        for dir_path in [self.inference_dir, self.gt_dir, self.hook_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 설정 및 모델 로드
        self.cfg = Config.fromfile(config_path)
        mmengine.registry.init_default_scope(self.cfg.get('default_scope', 'mmdet'))
        
        # 모델 초기화
        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.model.eval()
        
        # 데이터셋 로드
        val_dataset_cfg = self.cfg.val_dataloader.dataset
        self.dataset = DATASETS.build(val_dataset_cfg)
        self.dataset.full_init()
        
        # Visualizer 초기화
        self.visualizer = DetLocalVisualizer()
        self.visualizer.dataset_meta = self.dataset.metainfo
        
        # CMNeXt Hook 활성화를 위한 설정
        self._setup_cmnext_hook()
        
        print(f"평가 준비 완료:")
        print(f"  - 모델: {checkpoint_path}")
        print(f"  - 데이터셋: {len(self.dataset)}개 샘플")
        print(f"  - 클래스: {self.dataset.metainfo['classes']}")
        print(f"  - 출력 디렉토리: {output_dir}")
    
    def _setup_cmnext_hook(self):
        """CMNeXt Visualization Hook 설정"""
        try:
            # 모델에서 backbone 접근
            backbone = self.model.backbone
            if hasattr(backbone, 'cmnext_model') and hasattr(backbone.cmnext_model, 'backbone'):
                cmnext_backbone = backbone.cmnext_model.backbone
                # visualization_buffer 초기화
                if not hasattr(cmnext_backbone, 'visualization_buffer'):
                    cmnext_backbone.visualization_buffer = {}
                print("CMNeXt Visualization Hook 설정 완료")
            else:
                print("Warning: CMNeXt backbone을 찾을 수 없습니다.")
        except Exception as e:
            print(f"Warning: CMNeXt Hook 설정 실패: {e}")
    
    def _save_hook_visualizations(self, sample_idx: int, img_name: str):
        """CMNeXt Hook에서 생성된 시각화 정보 저장"""
        try:
            backbone = self.model.backbone
            if hasattr(backbone, 'cmnext_model') and hasattr(backbone.cmnext_model, 'backbone'):
                cmnext_backbone = backbone.cmnext_model.backbone
                if hasattr(cmnext_backbone, 'visualization_buffer') and cmnext_backbone.visualization_buffer:
                    buffer = cmnext_backbone.visualization_buffer
                    
                    # Hook 정보를 JSON으로 저장
                    hook_info = {}
                    for stage, tensors in buffer.items():
                        stage_info = {}
                        
                        # Attention weights 정보
                        if tensors.get('attention_weights') is not None:
                            att_weights = tensors['attention_weights']
                            if isinstance(att_weights, torch.Tensor):
                                # 평균 기여도 계산
                                avg_contributions = att_weights[0].mean(dim=[1, 2]).cpu().numpy()
                                stage_info['attention_contributions'] = {
                                    f'modal_{i}': float(contrib) for i, contrib in enumerate(avg_contributions)
                                }
                                
                                # Attention map 이미지 저장
                                self._save_attention_map(att_weights, stage, img_name)
                        
                        # Feature map 통계
                        for feat_name in ['rgb_feature', 'fused_aux_feature', 'final_fused_feature']:
                            if tensors.get(feat_name) is not None:
                                feat_tensor = tensors[feat_name]
                                if isinstance(feat_tensor, torch.Tensor):
                                    feat_stats = {
                                        'shape': list(feat_tensor.shape),
                                        'mean': float(feat_tensor.mean().cpu()),
                                        'std': float(feat_tensor.std().cpu()),
                                        'min': float(feat_tensor.min().cpu()),
                                        'max': float(feat_tensor.max().cpu())
                                    }
                                    stage_info[f'{feat_name}_stats'] = feat_stats
                                    
                                    # Feature map 시각화 저장
                                    self._save_feature_map(feat_tensor, f"{stage}_{feat_name}", img_name)
                        
                        hook_info[stage] = stage_info
                    
                    # JSON 파일로 저장
                    hook_json_path = os.path.join(self.hook_dir, f"{img_name}_hook_info.json")
                    with open(hook_json_path, 'w') as f:
                        json.dump(hook_info, f, indent=2)
                    
                    # Buffer 클리어
                    cmnext_backbone.visualization_buffer.clear()
                    
                    return hook_info
        except Exception as e:
            print(f"Hook 시각화 저장 실패: {e}")
        
        return {}
    
    def _save_attention_map(self, attention_weights: torch.Tensor, stage: str, img_name: str):
        """Attention map을 이미지로 저장"""
        try:
            # (B, num_modals, H, W) -> (num_modals, H, W)
            att_map = attention_weights[0].cpu().numpy()
            num_modals, h, w = att_map.shape
            
            # RGB 채널에 매핑하여 시각화
            rgb_map = np.zeros((h, w, 3), dtype=np.float32)
            if num_modals >= 1: rgb_map[:, :, 0] = att_map[0]  # R
            if num_modals >= 2: rgb_map[:, :, 1] = att_map[1]  # G  
            if num_modals >= 3: rgb_map[:, :, 2] = att_map[2]  # B
            
            # 0-255 범위로 변환
            rgb_map = (rgb_map * 255).astype(np.uint8)
            
            # 저장
            att_path = os.path.join(self.hook_dir, f"{img_name}_{stage}_attention_map.png")
            cv2.imwrite(att_path, cv2.cvtColor(rgb_map, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Attention map 저장 실패: {e}")
    
    def _save_feature_map(self, feature_tensor: torch.Tensor, feat_name: str, img_name: str):
        """Feature map을 그리드 형태로 시각화하여 저장"""
        try:
            # (B, C, H, W) -> (C, H, W)
            if feature_tensor.dim() == 4:
                feat_map = feature_tensor[0].cpu().numpy()
            else:
                feat_map = feature_tensor.cpu().numpy()
            
            C, H, W = feat_map.shape
            
            # 채널 수에 따라 그리드 크기 결정
            ncols = min(8, C)  # 최대 8열
            nrows = (C + ncols - 1) // ncols
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))
            if C == 1:
                axes = [axes]
            elif nrows == 1:
                axes = axes if isinstance(axes, np.ndarray) else [axes]
            else:
                axes = axes.flatten()
            
            for i in range(C):
                channel = feat_map[i]
                # 정규화
                if channel.max() > channel.min():
                    channel = (channel - channel.min()) / (channel.max() - channel.min())
                
                ax = axes[i] if i < len(axes) else axes[-1]
                ax.imshow(channel, cmap='jet')
                ax.axis('off')
                ax.set_title(f'Ch{i}', fontsize=8)
            
            # 빈 subplot 숨기기
            for j in range(C, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            feat_path = os.path.join(self.hook_dir, f"{img_name}_{feat_name}_feature_map.png")
            plt.savefig(feat_path, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Feature map 저장 실패: {e}")
    
    def _draw_bboxes_on_image(self, img: np.ndarray, bboxes: np.ndarray, 
                             labels: np.ndarray, scores: np.ndarray = None, 
                             class_names: List[str] = None) -> np.ndarray:
        """이미지에 바운딩 박스 그리기"""
        img_with_boxes = img.copy()
        
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # bbox 형식 확인 및 변환
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox.astype(int)
            elif len(bbox) == 5:  # xywh + confidence
                x1, y1, w, h, conf = bbox.astype(int)
                x2, y2 = x1 + w, y1 + h
            else:
                print(f"Warning: Unexpected bbox format with {len(bbox)} values: {bbox}")
                continue
            
            # 색상 설정 (클래스별로 다른 색상)
            color = plt.cm.Set1(label / max(len(class_names) if class_names else 10, 1))
            color = tuple(int(c * 255) for c in color[:3])
            
            # 바운딩 박스 그리기
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # 레이블 텍스트
            if class_names and label < len(class_names):
                class_name = class_names[label]
            else:
                class_name = f'class_{label}'
            
            if scores is not None and i < len(scores):
                text = f'{class_name}: {scores[i]:.2f}'
            else:
                text = class_name
            
            # 텍스트 배경
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_with_boxes, (x1, y1-text_h-5), (x1+text_w, y1), color, -1)
            cv2.putText(img_with_boxes, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return img_with_boxes
    
    def evaluate_sample(self, sample_idx: int, save_visualizations: bool = True) -> Dict[str, Any]:
        """단일 샘플 평가"""
        # 데이터 로드
        data_sample = self.dataset[sample_idx]
        
        # 이미지 정보 추출 - 수정된 방식
        data_samples = data_sample['data_samples']
        
        # RGB 이미지 경로 가져오기 - 수정된 로직
        try:
            # 1. data_samples의 metainfo에서 img_path 확인
            if hasattr(data_samples, 'metainfo') and 'img_path' in data_samples.metainfo:
                img_paths = data_samples.metainfo['img_path']
                if isinstance(img_paths, list) and len(img_paths) > 0:
                    img_path = img_paths[0]  # RGB 이미지 (첫 번째)
                else:
                    img_path = img_paths
            # 2. modality_paths에서 RGB 경로 확인
            elif hasattr(data_samples, 'metainfo') and 'modality_paths' in data_samples.metainfo:
                modality_paths = data_samples.metainfo['modality_paths']
                if isinstance(modality_paths, dict) and 'rgb' in modality_paths:
                    img_path = modality_paths['rgb']
                else:
                    raise KeyError("No RGB path in modality_paths")
            # 3. 원본 data_sample에서 직접 확인 (fallback)
            elif 'inputs' in data_sample:
                inputs = data_sample['inputs']
                if isinstance(inputs, list) and len(inputs) > 0:
                    # inputs가 텐서 리스트인 경우, 원본 데이터에서 경로 추출
                    # 이 경우 원본 데이터셋에서 직접 경로를 가져와야 함
                    raw_data = self.dataset.data_list[sample_idx]
                    if 'modality_paths' in raw_data and 'rgb' in raw_data['modality_paths']:
                        img_path = raw_data['modality_paths']['rgb']
                    else:
                        raise KeyError("No RGB path in raw data")
                else:
                    raise KeyError("Invalid inputs format")
            else:
                raise KeyError("No image path found in any location")
            
            # 텐서인 경우 처리 불가
            if hasattr(img_path, 'shape'):  # 텐서인 경우
                print(f"Warning: 이미지 경로가 텐서입니다. 스킵합니다.")
                return {'num_predictions': 0, 'num_gt': 0, 'classes': []}
            
            # 문자열이 아닌 경우 변환
            if not isinstance(img_path, str):
                img_path = str(img_path)
                
        except Exception as e:
            print(f"이미지 경로 추출 실패: {e}")
            print(f"data_sample keys: {list(data_sample.keys()) if isinstance(data_sample, dict) else 'Not a dict'}")
            if hasattr(data_samples, 'metainfo'):
                print(f"data_samples.metainfo keys: {list(data_samples.metainfo.keys())}")
            return {'num_predictions': 0, 'num_gt': 0, 'classes': []}  # 이미지 경로를 찾을 수 없으면 빈 결과 반환
        
        img_name = Path(img_path).stem
        
        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: 이미지를 로드할 수 없습니다: {img_path}")
            return {'num_predictions': 0, 'num_gt': 0, 'classes': []}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ground Truth 정보 - 수정된 방식
        gt_bboxes = np.array([])
        gt_labels = np.array([])
        
        if hasattr(data_samples, 'gt_instances') and data_samples.gt_instances is not None:
            gt_instances = data_samples.gt_instances
            if hasattr(gt_instances, 'bboxes') and len(gt_instances.bboxes) > 0:
                gt_bboxes = gt_instances.bboxes.cpu().numpy() if hasattr(gt_instances.bboxes, 'cpu') else gt_instances.bboxes.numpy()
            if hasattr(gt_instances, 'labels') and len(gt_instances.labels) > 0:
                gt_labels = gt_instances.labels.cpu().numpy() if hasattr(gt_instances.labels, 'cpu') else gt_instances.labels.numpy()
        
        # 추론 수행 - 이미지 경로를 직접 사용
        with torch.no_grad():
            # 이미지 경로를 직접 사용하여 추론
            result = inference_detector(self.model, img_path)
        
        # 예측 결과 추출
        pred_instances = result.pred_instances
        pred_bboxes = pred_instances.bboxes.cpu().numpy() if len(pred_instances) > 0 else np.array([])
        pred_labels = pred_instances.labels.cpu().numpy() if len(pred_instances) > 0 else np.array([])
        pred_scores = pred_instances.scores.cpu().numpy() if len(pred_instances) > 0 else np.array([])
        
        # Hook 정보 저장
        hook_info = {}
        if save_visualizations:
            hook_info = self._save_hook_visualizations(sample_idx, img_name)
        
        # 시각화 저장
        if save_visualizations:
            class_names = self.dataset.metainfo['classes']
            
            # 1. 원본 이미지 저장
            original_path = os.path.join(self.inference_dir, f"{img_name}_original.jpg")
            cv2.imwrite(original_path, img)
            
            # 2. GT 시각화
            if len(gt_bboxes) > 0:
                gt_img = self._draw_bboxes_on_image(img_rgb, gt_bboxes, gt_labels, 
                                                  class_names=class_names)
                gt_path = os.path.join(self.gt_dir, f"{img_name}_gt.jpg")
                cv2.imwrite(gt_path, cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
            
            # 3. 예측 결과 시각화
            if len(pred_bboxes) > 0:
                pred_img = self._draw_bboxes_on_image(img_rgb, pred_bboxes, pred_labels, 
                                                    pred_scores, class_names=class_names)
                pred_path = os.path.join(self.inference_dir, f"{img_name}_prediction.jpg")
                cv2.imwrite(pred_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
            
            # 4. GT + 예측 결과 비교 이미지
            comparison_img = np.hstack([
                self._draw_bboxes_on_image(img_rgb, gt_bboxes, gt_labels, class_names=class_names),
                self._draw_bboxes_on_image(img_rgb, pred_bboxes, pred_labels, pred_scores, class_names=class_names)
            ])
            comparison_path = os.path.join(self.inference_dir, f"{img_name}_comparison.jpg")
            cv2.imwrite(comparison_path, cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))
        
        # 결과 정리
        sample_result = {
            'sample_idx': sample_idx,
            'img_name': img_name,
            'img_path': img_path,
            'gt_bboxes': gt_bboxes.tolist() if len(gt_bboxes) > 0 else [],
            'gt_labels': gt_labels.tolist() if len(gt_labels) > 0 else [],
            'pred_bboxes': pred_bboxes.tolist() if len(pred_bboxes) > 0 else [],
            'pred_labels': pred_labels.tolist() if len(pred_labels) > 0 else [],
            'pred_scores': pred_scores.tolist() if len(pred_scores) > 0 else [],
            'num_gt': len(gt_bboxes),
            'num_pred': len(pred_bboxes),
            'num_predictions': len(pred_bboxes),  # 호환성을 위해 추가
            'hook_info': hook_info
        }
        
        return sample_result
    
    def evaluate_dataset(self, num_samples: int = -1, save_visualizations: bool = True) -> Dict[str, Any]:
        """전체 데이터셋 평가"""
        print("데이터셋 평가 시작...")
        
        # 샘플 수 결정
        total_samples = len(self.dataset) if num_samples < 0 else min(num_samples, len(self.dataset))
        sample_indices = list(range(total_samples))
        
        print(f"전체 데이터셋 크기: {len(self.dataset)}")
        print(f"요청된 샘플 수: {num_samples}")
        print(f"실제 평가할 샘플 수: {total_samples}")
        
        if num_samples < 0:
            print("✅ 전체 데이터셋 평가 모드")
        
        # 평가 결과 저장
        all_results = []
        evaluation_stats = {
            'total_samples': total_samples,
            'config_path': self.config_path,
            'checkpoint_path': self.checkpoint_path,
            'evaluation_time': datetime.now().isoformat(),
            'class_names': self.dataset.metainfo['classes']
        }
        
        # 각 샘플 평가
        for i, idx in enumerate(sample_indices):
            if i % 10 == 0:
                print(f"진행률: {i+1}/{total_samples}")
            
            try:
                sample_result = self.evaluate_sample(idx, save_visualizations)
                all_results.append(sample_result)
            except Exception as e:
                import traceback
                print(f"샘플 {idx} 평가 실패: {e}")
                print(f"상세 오류: {traceback.format_exc()}")
                continue
        
        # 통계 계산
        total_gt = sum(result['num_gt'] for result in all_results)
        total_pred = sum(result['num_predictions'] for result in all_results)
        avg_gt_per_image = total_gt / len(all_results) if all_results else 0
        avg_pred_per_image = total_pred / len(all_results) if all_results else 0
        
        evaluation_stats.update({
            'processed_samples': len(all_results),
            'total_gt_objects': total_gt,
            'total_pred_objects': total_pred,
            'avg_gt_per_image': avg_gt_per_image,
            'avg_pred_per_image': avg_pred_per_image
        })
        
        # 결과 저장
        results_file = os.path.join(self.results_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'statistics': evaluation_stats,
                'sample_results': all_results
            }, f, indent=2)
        
        # 통계 요약 저장
        summary_file = os.path.join(self.results_dir, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(evaluation_stats, f, indent=2)
        
        print(f"\n평가 완료!")
        print(f"처리된 샘플: {len(all_results)}/{total_samples}")
        print(f"총 GT 객체: {total_gt}")
        print(f"총 예측 객체: {total_pred}")
        print(f"이미지당 평균 GT: {avg_gt_per_image:.2f}")
        print(f"이미지당 평균 예측: {avg_pred_per_image:.2f}")
        print(f"결과 저장: {self.results_dir}")
        
        return evaluation_stats


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--config', required=True, help='Model config file path')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint file path')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=-1, 
                       help='Number of samples to evaluate (-1 for all)')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--no-visualization', action='store_true', 
                       help='Skip saving visualization images')
    
    args = parser.parse_args()
    
    # 평가 실행
    evaluator = ComprehensiveEvaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device
    )
    
    results = evaluator.evaluate_dataset(
        num_samples=args.num_samples,
        save_visualizations=not args.no_visualization
    )
    
    print(f"\n=== 평가 완료 ===")
    print(f"결과 디렉토리: {args.output_dir}")
    print(f"- inference_images/: 예측 결과 이미지")
    print(f"- gt_visualizations/: Ground Truth 시각화")
    print(f"- hook_visualizations/: CMNeXt Hook 시각화")
    print(f"- evaluation_results/: 평가 결과 JSON")


if __name__ == '__main__':
    main()
