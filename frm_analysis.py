#!/usr/bin/env python3
"""
FRM (Feature Rectify Module) 모듈 선택 분석 스크립트
CMNeXtPSP 모델에서 FRM 모듈이 어떻게 선택되는지 분석합니다.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import mmengine
from mmdet.apis import init_detector
from mmdet.registry import DATASETS
from collections import defaultdict, Counter

# 사용자의 커스텀 모듈 등록
from mcdet import *

class FRMAnalyzer:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.cfg = mmengine.Config.fromfile(config_path)
        mmengine.registry.init_default_scope(self.cfg.get('default_scope', 'mmdet'))
        self.model = init_detector(config_path, checkpoint_path, device=device, cfg_options={'strict': False})
        
        # FRM 분석 데이터 저장
        self.frm_data = {}
        self.attention_weights = {}
        self.channel_weights = {}
        self.spatial_weights = {}
        
        # TokenSelect 분석 데이터 저장
        self.tokenselect_data = {}
        self.modality_selection_counts = defaultdict(lambda: defaultdict(int))
        self.selection_scores = defaultdict(list)
        
        # 훅 등록
        self.register_frm_hooks()
        self.register_tokenselect_hooks()
    
    def register_frm_hooks(self):
        """FRM 모듈 분석을 위한 훅 등록"""
        
        def frm_forward_hook(module, input, output):
            """FRM forward 훅"""
            stage_name = getattr(module, '_stage_name', 'unknown')
            
            if stage_name not in self.frm_data:
                self.frm_data[stage_name] = {
                    'inputs': [],
                    'outputs': [],
                    'channel_weights': [],
                    'spatial_weights': [],
                    'lambda_c': getattr(module, 'lambda_c', 0.5),
                    'lambda_s': getattr(module, 'lambda_s', 0.5)
                }
            
            # 입력 분석
            if isinstance(input, (list, tuple)) and len(input) >= 2:
                x1, x2 = input[0], input[1]
                self.frm_data[stage_name]['inputs'].append({
                    'x1_shape': list(x1.shape),
                    'x2_shape': list(x2.shape),
                    'x1_mean': x1.mean().item(),
                    'x2_mean': x2.mean().item(),
                    'x1_std': x1.std().item(),
                    'x2_std': x2.std().item()
                })
            
            # 출력 분석
            if isinstance(output, (list, tuple)) and len(output) >= 2:
                out_x1, out_x2 = output[0], output[1]
                self.frm_data[stage_name]['outputs'].append({
                    'out_x1_shape': list(out_x1.shape),
                    'out_x2_shape': list(out_x2.shape),
                    'out_x1_mean': out_x1.mean().item(),
                    'out_x2_mean': out_x2.mean().item(),
                    'out_x1_std': out_x1.std().item(),
                    'out_x2_std': out_x2.std().item()
                })
            
            # 채널 가중치 분석
            if hasattr(module, 'channel_weights'):
                try:
                    with torch.no_grad():
                        if isinstance(input, (list, tuple)) and len(input) >= 2:
                            x1, x2 = input[0], input[1]
                            ch_weights = module.channel_weights(x1, x2)
                            self.frm_data[stage_name]['channel_weights'].append({
                                'weight_0_mean': ch_weights[0].mean().item(),
                                'weight_1_mean': ch_weights[1].mean().item(),
                                'weight_0_std': ch_weights[0].std().item(),
                                'weight_1_std': ch_weights[1].std().item(),
                                'weight_0_max': ch_weights[0].max().item(),
                                'weight_1_max': ch_weights[1].max().item(),
                                'weight_0_min': ch_weights[0].min().item(),
                                'weight_1_min': ch_weights[1].min().item()
                            })
                except Exception as e:
                    print(f"채널 가중치 분석 중 오류: {e}")
            
            # 공간 가중치 분석
            if hasattr(module, 'spatial_weights'):
                try:
                    with torch.no_grad():
                        if isinstance(input, (list, tuple)) and len(input) >= 2:
                            x1, x2 = input[0], input[1]
                            sp_weights = module.spatial_weights(x1, x2)
                            self.frm_data[stage_name]['spatial_weights'].append({
                                'weight_0_mean': sp_weights[0].mean().item(),
                                'weight_1_mean': sp_weights[1].mean().item(),
                                'weight_0_std': sp_weights[0].std().item(),
                                'weight_1_std': sp_weights[1].std().item(),
                                'weight_0_max': sp_weights[0].max().item(),
                                'weight_1_max': sp_weights[1].max().item(),
                                'weight_0_min': sp_weights[0].min().item(),
                                'weight_1_min': sp_weights[1].min().item()
                            })
                except Exception as e:
                    print(f"공간 가중치 분석 중 오류: {e}")
        
        # 모델의 모든 FRM 모듈에 훅 등록
        for name, module in self.model.named_modules():
            if 'FRM' in str(module.__class__) or 'FeatureRectifyModule' in str(module.__class__):
                # 스테이지 이름 추출
                parts = name.split('.')
                stage_name = 'unknown'
                for part in parts:
                    if 'stage' in part.lower() or part.isdigit():
                        stage_name = f"stage_{part}"
                        break
                
                module._stage_name = stage_name
                module.register_forward_hook(frm_forward_hook)
                print(f"FRM 훅 등록: {name} -> {stage_name}")
    
    def register_tokenselect_hooks(self):
        """TokenSelect 모듈 분석을 위한 훅 등록"""
        
        def tokenselect_hook(module, input, output):
            """TokenSelect forward 훅"""
            stage_name = getattr(module, '_stage_name', 'unknown')
            
            if stage_name not in self.tokenselect_data:
                self.tokenselect_data[stage_name] = {
                    'selections': [],
                    'scores': []
                }
            
            # 입력 분석 (x_ext: List[Tensor])
            if isinstance(input, (list, tuple)) and len(input) > 0:
                x_ext = input[0] if isinstance(input[0], list) else input
                
                # 출력 분석
                if isinstance(output, (list, tuple)) and len(output) >= 3:
                    x_f, x_scores, x_f_indices = output[0], output[1], output[2]
                    
                    # 선택된 모달리티 인덱스 분석
                    if isinstance(x_f_indices, torch.Tensor):
                        selected_indices = x_f_indices.cpu().numpy()
                        unique_indices, counts = np.unique(selected_indices, return_counts=True)
                        
                        selection_info = {
                            'selected_indices': selected_indices.tolist(),
                            'unique_selections': unique_indices.tolist(),
                            'selection_counts': counts.tolist(),
                            'most_selected': int(unique_indices[np.argmax(counts)]),
                            'selection_entropy': self.calculate_entropy(counts)
                        }
                        
                        # 모달리티별 선택 횟수 기록
                        for idx, count in zip(unique_indices, counts):
                            self.modality_selection_counts[stage_name][int(idx)] += int(count)
                        
                        self.tokenselect_data[stage_name]['selections'].append(selection_info)
                    
                    # 점수 분석
                    if isinstance(x_scores, list):
                        scores_info = []
                        for i, score in enumerate(x_scores):
                            if isinstance(score, torch.Tensor):
                                scores_info.append({
                                    'modality_idx': i,
                                    'mean_score': score.mean().item(),
                                    'std_score': score.std().item(),
                                    'max_score': score.max().item(),
                                    'min_score': score.min().item()
                                })
                        self.tokenselect_data[stage_name]['scores'].append(scores_info)
                        
                        # 점수를 리스트로 저장
                        score_values = [s['mean_score'] for s in scores_info]
                        self.selection_scores[stage_name].append(score_values)
        
        # 모델의 TokenSelect 관련 모듈에 훅 등록
        for name, module in self.model.named_modules():
            if 'tokenselect' in name.lower() or 'TokenSelect' in str(module.__class__):
                # 스테이지 이름 추출
                parts = name.split('.')
                stage_name = 'unknown'
                for part in parts:
                    if 'stage' in part.lower() or part.isdigit():
                        stage_name = f"stage_{part}"
                        break
                
                module._stage_name = stage_name
                module.register_forward_hook(tokenselect_hook)
                print(f"TokenSelect 훅 등록: {name} -> {stage_name}")
    
    def calculate_entropy(self, counts):
        """선택 분포의 엔트로피 계산"""
        total = sum(counts)
        if total == 0:
            return 0
        probs = [c / total for c in counts]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy
    
    def run_analysis(self, num_samples: int = 50):
        """FRM 분석 실행"""
        print(f"FRM 분석 시작: {num_samples}개 샘플")
        
        # 데이터셋 로드
        val_dataset_cfg = self.cfg.val_dataloader.dataset
        dataset = DATASETS.build(val_dataset_cfg)
        dataset.full_init()
        
        total_samples = len(dataset) if num_samples < 0 else min(num_samples, len(dataset))
        
        self.model.eval()
        
        with torch.no_grad():
            for i in range(total_samples):
                if i % 10 == 0:
                    print(f"진행률: {i+1}/{total_samples}")
                
                # 데이터 로드
                data = dataset[i]
                
                # 추론 실행
                batched_data = { 
                    'inputs': [[item] for item in data['inputs']], 
                    'data_samples': [data['data_samples']] 
                }
                processed_data = self.model.data_preprocessor(batched_data, training=False)
                predictions = self.model.forward(**processed_data, mode='predict')
        
        print("FRM 분석 완료")
        return self.frm_data
    
    def analyze_channel_weights(self, output_dir: str):
        """채널 가중치 분석"""
        print("채널 가중치 분석 중...")
        
        if not self.frm_data:
            print("FRM 데이터가 없습니다. 먼저 run_analysis()를 실행하세요.")
            return
        
        # 각 스테이지별 채널 가중치 분석
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FRM Channel Weights Analysis', fontsize=16)
        
        stage_names = list(self.frm_data.keys())
        
        for idx, stage_name in enumerate(stage_names[:4]):  # 최대 4개 스테이지
            if stage_name not in self.frm_data:
                continue
            
            data = self.frm_data[stage_name]
            if not data['channel_weights']:
                continue
            
            row, col = idx // 2, idx % 2
            
            # 채널 가중치 0과 1의 분포
            weights_0 = [w['weight_0_mean'] for w in data['channel_weights']]
            weights_1 = [w['weight_1_mean'] for w in data['channel_weights']]
            
            axes[row, col].hist(weights_0, alpha=0.7, label='Channel Weight 0', bins=20)
            axes[row, col].hist(weights_1, alpha=0.7, label='Channel Weight 1', bins=20)
            axes[row, col].set_title(f'{stage_name} - Channel Weights Distribution')
            axes[row, col].set_xlabel('Weight Value')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frm_channel_weights_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 채널 가중치 통계 저장
        channel_stats = {}
        for stage_name, data in self.frm_data.items():
            if data['channel_weights']:
                weights_0 = [w['weight_0_mean'] for w in data['channel_weights']]
                weights_1 = [w['weight_1_mean'] for w in data['channel_weights']]
                
                channel_stats[stage_name] = {
                    'weight_0_stats': {
                        'mean': np.mean(weights_0),
                        'std': np.std(weights_0),
                        'min': np.min(weights_0),
                        'max': np.max(weights_0)
                    },
                    'weight_1_stats': {
                        'mean': np.mean(weights_1),
                        'std': np.std(weights_1),
                        'min': np.min(weights_1),
                        'max': np.max(weights_1)
                    },
                    'lambda_c': data['lambda_c'],
                    'lambda_s': data['lambda_s']
                }
        
        with open(os.path.join(output_dir, 'frm_channel_weights_stats.json'), 'w') as f:
            json.dump(channel_stats, f, indent=4)
        
        print(f"채널 가중치 분석 결과 저장: {output_dir}/frm_channel_weights_analysis.png")
        return channel_stats
    
    def analyze_spatial_weights(self, output_dir: str):
        """공간 가중치 분석"""
        print("공간 가중치 분석 중...")
        
        if not self.frm_data:
            print("FRM 데이터가 없습니다. 먼저 run_analysis()를 실행하세요.")
            return
        
        # 각 스테이지별 공간 가중치 분석
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FRM Spatial Weights Analysis', fontsize=16)
        
        stage_names = list(self.frm_data.keys())
        
        for idx, stage_name in enumerate(stage_names[:4]):  # 최대 4개 스테이지
            if stage_name not in self.frm_data:
                continue
            
            data = self.frm_data[stage_name]
            if not data['spatial_weights']:
                continue
            
            row, col = idx // 2, idx % 2
            
            # 공간 가중치 0과 1의 분포
            weights_0 = [w['weight_0_mean'] for w in data['spatial_weights']]
            weights_1 = [w['weight_1_mean'] for w in data['spatial_weights']]
            
            axes[row, col].hist(weights_0, alpha=0.7, label='Spatial Weight 0', bins=20)
            axes[row, col].hist(weights_1, alpha=0.7, label='Spatial Weight 1', bins=20)
            axes[row, col].set_title(f'{stage_name} - Spatial Weights Distribution')
            axes[row, col].set_xlabel('Weight Value')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frm_spatial_weights_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 공간 가중치 통계 저장
        spatial_stats = {}
        for stage_name, data in self.frm_data.items():
            if data['spatial_weights']:
                weights_0 = [w['weight_0_mean'] for w in data['spatial_weights']]
                weights_1 = [w['weight_1_mean'] for w in data['spatial_weights']]
                
                spatial_stats[stage_name] = {
                    'weight_0_stats': {
                        'mean': np.mean(weights_0),
                        'std': np.std(weights_0),
                        'min': np.min(weights_0),
                        'max': np.max(weights_0)
                    },
                    'weight_1_stats': {
                        'mean': np.mean(weights_1),
                        'std': np.std(weights_1),
                        'min': np.min(weights_1),
                        'max': np.max(weights_1)
                    }
                }
        
        with open(os.path.join(output_dir, 'frm_spatial_weights_stats.json'), 'w') as f:
            json.dump(spatial_stats, f, indent=4)
        
        print(f"공간 가중치 분석 결과 저장: {output_dir}/frm_spatial_weights_analysis.png")
        return spatial_stats
    
    def analyze_feature_rectification(self, output_dir: str):
        """특징 정제 효과 분석"""
        print("특징 정제 효과 분석 중...")
        
        if not self.frm_data:
            print("FRM 데이터가 없습니다. 먼저 run_analysis()를 실행하세요.")
            return
        
        # 입력과 출력의 변화 분석
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FRM Feature Rectification Effect', fontsize=16)
        
        stage_names = list(self.frm_data.keys())
        
        for idx, stage_name in enumerate(stage_names[:4]):
            if stage_name not in self.frm_data:
                continue
            
            data = self.frm_data[stage_name]
            if not data['inputs'] or not data['outputs']:
                continue
            
            row, col = idx // 2, idx % 2
            
            # 입력과 출력의 평균값 비교
            input_means_x1 = [inp['x1_mean'] for inp in data['inputs']]
            input_means_x2 = [inp['x2_mean'] for inp in data['inputs']]
            output_means_x1 = [out['out_x1_mean'] for out in data['outputs']]
            output_means_x2 = [out['out_x2_mean'] for out in data['outputs']]
            
            x = range(len(input_means_x1))
            axes[row, col].plot(x, input_means_x1, 'b-', alpha=0.7, label='Input X1')
            axes[row, col].plot(x, input_means_x2, 'r-', alpha=0.7, label='Input X2')
            axes[row, col].plot(x, output_means_x1, 'b--', alpha=0.7, label='Output X1')
            axes[row, col].plot(x, output_means_x2, 'r--', alpha=0.7, label='Output X2')
            axes[row, col].set_title(f'{stage_name} - Input vs Output')
            axes[row, col].set_xlabel('Sample Index')
            axes[row, col].set_ylabel('Mean Value')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frm_rectification_effect.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 정제 효과 통계 저장
        rectification_stats = {}
        for stage_name, data in self.frm_data.items():
            if data['inputs'] and data['outputs']:
                input_x1_means = [inp['x1_mean'] for inp in data['inputs']]
                input_x2_means = [inp['x2_mean'] for inp in data['inputs']]
                output_x1_means = [out['out_x1_mean'] for out in data['outputs']]
                output_x2_means = [out['out_x2_mean'] for out in data['outputs']]
                
                rectification_stats[stage_name] = {
                    'input_x1_mean': np.mean(input_x1_means),
                    'input_x2_mean': np.mean(input_x2_means),
                    'output_x1_mean': np.mean(output_x1_means),
                    'output_x2_mean': np.mean(output_x2_means),
                    'x1_improvement': np.mean(output_x1_means) - np.mean(input_x1_means),
                    'x2_improvement': np.mean(output_x2_means) - np.mean(input_x2_means)
                }
        
        with open(os.path.join(output_dir, 'frm_rectification_stats.json'), 'w') as f:
            json.dump(rectification_stats, f, indent=4)
        
        print(f"특징 정제 효과 분석 결과 저장: {output_dir}/frm_rectification_effect.png")
        return rectification_stats
    
    def analyze_tokenselect_selection(self, output_dir: str):
        """TokenSelect 모달리티 선택 분석"""
        print("TokenSelect 모달리티 선택 분석 중...")
        
        if not self.tokenselect_data:
            print("TokenSelect 데이터가 없습니다.")
            return
        
        # 각 스테이지별 선택 패턴 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('TokenSelect Modality Selection Patterns', fontsize=16)
        
        stage_names = list(self.tokenselect_data.keys())
        
        for idx, stage_name in enumerate(stage_names[:4]):  # 최대 4개 스테이지
            if stage_name not in self.tokenselect_data:
                continue
            
            data = self.tokenselect_data[stage_name]
            if not data['selections']:
                continue
            
            row, col = idx // 2, idx % 2
            
            # 선택된 모달리티 분포
            all_selections = []
            for selection in data['selections']:
                all_selections.extend(selection['selected_indices'])
            
            if all_selections:
                unique, counts = np.unique(all_selections, return_counts=True)
                axes[row, col].bar(unique, counts, alpha=0.7)
                axes[row, col].set_title(f'{stage_name} - Modality Selection Distribution')
                axes[row, col].set_xlabel('Modality Index')
                axes[row, col].set_ylabel('Selection Count')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tokenselect_selection_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 선택 패턴 통계 저장
        selection_stats = {}
        for stage_name, data in self.tokenselect_data.items():
            if data['selections']:
                all_selections = []
                entropies = []
                for selection in data['selections']:
                    all_selections.extend(selection['selected_indices'])
                    entropies.append(selection['selection_entropy'])
                
                if all_selections:
                    unique, counts = np.unique(all_selections, return_counts=True)
                    selection_stats[stage_name] = {
                        'total_selections': len(all_selections),
                        'unique_modalities': len(unique),
                        'most_selected_modality': int(unique[np.argmax(counts)]),
                        'selection_counts': dict(zip(unique.tolist(), counts.tolist())),
                        'avg_entropy': np.mean(entropies),
                        'std_entropy': np.std(entropies)
                    }
        
        with open(os.path.join(output_dir, 'tokenselect_selection_stats.json'), 'w') as f:
            json.dump(selection_stats, f, indent=4)
        
        print(f"TokenSelect 선택 패턴 분석 결과 저장: {output_dir}/tokenselect_selection_patterns.png")
        return selection_stats
    
    def analyze_tokenselect_scores(self, output_dir: str):
        """TokenSelect 선택 점수 분석"""
        print("TokenSelect 선택 점수 분석 중...")
        
        if not self.selection_scores:
            print("선택 점수 데이터가 없습니다.")
            return
        
        # 각 스테이지별 점수 분포 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('TokenSelect Modality Selection Scores', fontsize=16)
        
        stage_names = list(self.selection_scores.keys())
        
        for idx, stage_name in enumerate(stage_names[:4]):
            if stage_name not in self.selection_scores:
                continue
            
            scores_data = self.selection_scores[stage_name]
            if not scores_data:
                continue
            
            row, col = idx // 2, idx % 2
            
            # 각 모달리티별 점수 분포
            num_modalities = len(scores_data[0]) if scores_data else 0
            for modality_idx in range(num_modalities):
                modality_scores = [sample[modality_idx] for sample in scores_data]
                axes[row, col].hist(modality_scores, alpha=0.6, label=f'Modality {modality_idx}', bins=20)
            
            axes[row, col].set_title(f'{stage_name} - Selection Scores Distribution')
            axes[row, col].set_xlabel('Score Value')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tokenselect_scores_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 점수 통계 저장
        score_stats = {}
        for stage_name, scores_data in self.selection_scores.items():
            if scores_data:
                num_modalities = len(scores_data[0])
                modality_stats = {}
                
                for modality_idx in range(num_modalities):
                    modality_scores = [sample[modality_idx] for sample in scores_data]
                    modality_stats[f'modality_{modality_idx}'] = {
                        'mean_score': np.mean(modality_scores),
                        'std_score': np.std(modality_scores),
                        'max_score': np.max(modality_scores),
                        'min_score': np.min(modality_scores)
                    }
                
                score_stats[stage_name] = {
                    'num_modalities': num_modalities,
                    'total_samples': len(scores_data),
                    'modality_stats': modality_stats
                }
        
        with open(os.path.join(output_dir, 'tokenselect_scores_stats.json'), 'w') as f:
            json.dump(score_stats, f, indent=4)
        
        print(f"TokenSelect 점수 분석 결과 저장: {output_dir}/tokenselect_scores_distribution.png")
        return score_stats
    
    def generate_summary_report(self, output_dir: str):
        """종합 분석 보고서 생성"""
        print("종합 분석 보고서 생성 중...")
        
        report = {
            'analysis_summary': {
                'total_stages': len(self.frm_data),
                'stages_analyzed': list(self.frm_data.keys()),
                'tokenselect_stages': len(self.tokenselect_data),
                'total_modality_selections': sum(sum(stage.values()) for stage in self.modality_selection_counts.values())
            },
            'stage_details': {}
        }
        
        for stage_name, data in self.frm_data.items():
            stage_report = {
                'lambda_c': data['lambda_c'],
                'lambda_s': data['lambda_s'],
                'total_samples': len(data['inputs']) if data['inputs'] else 0
            }
            
            if data['channel_weights']:
                ch_weights_0 = [w['weight_0_mean'] for w in data['channel_weights']]
                ch_weights_1 = [w['weight_1_mean'] for w in data['channel_weights']]
                stage_report['channel_weights'] = {
                    'weight_0_mean': np.mean(ch_weights_0),
                    'weight_1_mean': np.mean(ch_weights_1),
                    'weight_0_std': np.std(ch_weights_0),
                    'weight_1_std': np.std(ch_weights_1)
                }
            
            if data['spatial_weights']:
                sp_weights_0 = [w['weight_0_mean'] for w in data['spatial_weights']]
                sp_weights_1 = [w['weight_1_mean'] for w in data['spatial_weights']]
                stage_report['spatial_weights'] = {
                    'weight_0_mean': np.mean(sp_weights_0),
                    'weight_1_mean': np.mean(sp_weights_1),
                    'weight_0_std': np.std(sp_weights_0),
                    'weight_1_std': np.std(sp_weights_1)
                }
            
            report['stage_details'][stage_name] = stage_report
        
        with open(os.path.join(output_dir, 'frm_analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"종합 분석 보고서 저장: {output_dir}/frm_analysis_report.json")
        return report

def main():
    parser = argparse.ArgumentParser(description='FRM Module Selection Analysis')
    parser.add_argument('--config', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_heuristicalign_cmnextpsp_b2_rcnn_multiscale_v2_with_visualization/local-sejong2504_heuristicalign_cmnextpsp_rcnn_lr0.01_ep50_v2_with_visualization.py',
                       help='Model config file path')
    parser.add_argument('--checkpoint', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_heuristicalign_cmnextpsp_b2_rcnn_multiscale_v2_with_visualization/epoch_15.pth',
                       help='Model checkpoint file path')
    parser.add_argument('--output-dir', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/frm_analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # FRM 분석기 초기화
    analyzer = FRMAnalyzer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    print("=== FRM 모듈 선택 분석 시작 ===")
    
    # 1. FRM 분석 실행
    print("1. FRM 분석 실행...")
    frm_data = analyzer.run_analysis(num_samples=args.num_samples)
    
    # 2. 채널 가중치 분석
    print("2. 채널 가중치 분석...")
    channel_stats = analyzer.analyze_channel_weights(args.output_dir)
    
    # 3. 공간 가중치 분석
    print("3. 공간 가중치 분석...")
    spatial_stats = analyzer.analyze_spatial_weights(args.output_dir)
    
    # 4. 특징 정제 효과 분석
    print("4. 특징 정제 효과 분석...")
    rectification_stats = analyzer.analyze_feature_rectification(args.output_dir)
    
    # 5. TokenSelect 모달리티 선택 분석
    print("5. TokenSelect 모달리티 선택 분석...")
    tokenselect_selection_stats = analyzer.analyze_tokenselect_selection(args.output_dir)
    
    # 6. TokenSelect 선택 점수 분석
    print("6. TokenSelect 선택 점수 분석...")
    tokenselect_score_stats = analyzer.analyze_tokenselect_scores(args.output_dir)
    
    # 7. 종합 보고서 생성
    print("7. 종합 분석 보고서 생성...")
    report = analyzer.generate_summary_report(args.output_dir)
    
    print(f"\n=== FRM 분석 완료 ===")
    print(f"결과 디렉토리: {args.output_dir}")
    print(f"분석된 스테이지: {len(frm_data)}개")
    
    # 주요 결과 요약
    if report['stage_details']:
        print("\n=== 주요 결과 요약 ===")
        for stage_name, details in report['stage_details'].items():
            print(f"\n{stage_name}:")
            print(f"  - Lambda C: {details['lambda_c']}")
            print(f"  - Lambda S: {details['lambda_s']}")
            print(f"  - 분석 샘플 수: {details['total_samples']}")
            
            if 'channel_weights' in details:
                ch = details['channel_weights']
                print(f"  - 채널 가중치 0 평균: {ch['weight_0_mean']:.4f}")
                print(f"  - 채널 가중치 1 평균: {ch['weight_1_mean']:.4f}")

if __name__ == '__main__':
    main()
