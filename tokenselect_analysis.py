#!/usr/bin/env python3
"""
TokenSelect 모달리티 선택 패턴 분석 스크립트
CMNeXtPSP 모델에서 tokenselect2/tokenselect가 어떤 모달리티를 주로 선택하는지 분석합니다.
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

class TokenSelectAnalyzer:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        self.device = device
        self.cfg = mmengine.Config.fromfile(config_path)
        mmengine.registry.init_default_scope(self.cfg.get('default_scope', 'mmdet'))
        self.model = init_detector(config_path, checkpoint_path, device=device, cfg_options={'strict': False})
        
        # TokenSelect 분석 데이터 저장
        self.tokenselect_data = {}
        self.modality_selection_counts = defaultdict(lambda: defaultdict(int))
        self.selection_scores = defaultdict(list)
        self.attention_weights = defaultdict(list)
        
        # 훅 등록
        self.register_tokenselect_hooks()
    
    def register_tokenselect_hooks(self):
        """TokenSelect 관련 훅 등록"""
        
        def tokenselect_hook(module, input, output):
            """TokenSelect forward 훅"""
            stage_name = getattr(module, '_stage_name', 'unknown')
            
            if stage_name not in self.tokenselect_data:
                self.tokenselect_data[stage_name] = {
                    'selections': [],
                    'scores': [],
                    'attention_weights': []
                }
            
            # 입력 분석 (x_ext: List[Tensor])
            if isinstance(input, (list, tuple)) and len(input) > 0:
                x_ext = input[0] if isinstance(input[0], list) else input
                
                # 각 모달리티별 특성 분석
                modality_features = []
                for i, x in enumerate(x_ext):
                    if isinstance(x, torch.Tensor):
                        modality_features.append({
                            'modality_idx': i,
                            'mean': x.mean().item(),
                            'std': x.std().item(),
                            'max': x.max().item(),
                            'min': x.min().item(),
                            'shape': list(x.shape)
                        })
                
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
        """TokenSelect 분석 실행"""
        print(f"TokenSelect 분석 시작: {num_samples}개 샘플")
        
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
        
        print("TokenSelect 분석 완료")
        return self.tokenselect_data
    
    def analyze_modality_selection_patterns(self, output_dir: str):
        """모달리티 선택 패턴 분석"""
        print("모달리티 선택 패턴 분석 중...")
        
        if not self.tokenselect_data:
            print("TokenSelect 데이터가 없습니다. 먼저 run_analysis()를 실행하세요.")
            return
        
        # 각 스테이지별 선택 패턴 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Modality Selection Patterns by Stage', fontsize=16)
        
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
        plt.savefig(os.path.join(output_dir, 'modality_selection_patterns.png'), dpi=300, bbox_inches='tight')
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
        
        with open(os.path.join(output_dir, 'modality_selection_stats.json'), 'w') as f:
            json.dump(selection_stats, f, indent=4)
        
        print(f"모달리티 선택 패턴 분석 결과 저장: {output_dir}/modality_selection_patterns.png")
        return selection_stats
    
    def analyze_selection_scores(self, output_dir: str):
        """선택 점수 분석"""
        print("선택 점수 분석 중...")
        
        if not self.selection_scores:
            print("선택 점수 데이터가 없습니다.")
            return
        
        # 각 스테이지별 점수 분포 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Modality Selection Scores by Stage', fontsize=16)
        
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
        plt.savefig(os.path.join(output_dir, 'selection_scores_distribution.png'), dpi=300, bbox_inches='tight')
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
        
        with open(os.path.join(output_dir, 'selection_scores_stats.json'), 'w') as f:
            json.dump(score_stats, f, indent=4)
        
        print(f"선택 점수 분석 결과 저장: {output_dir}/selection_scores_distribution.png")
        return score_stats
    
    def analyze_selection_entropy(self, output_dir: str):
        """선택 엔트로피 분석 (다양성 측정)"""
        print("선택 엔트로피 분석 중...")
        
        if not self.tokenselect_data:
            print("TokenSelect 데이터가 없습니다.")
            return
        
        # 각 스테이지별 엔트로피 분포
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Selection Entropy by Stage (Diversity Analysis)', fontsize=16)
        
        stage_names = list(self.tokenselect_data.keys())
        
        for idx, stage_name in enumerate(stage_names[:4]):
            if stage_name not in self.tokenselect_data:
                continue
            
            data = self.tokenselect_data[stage_name]
            if not data['selections']:
                continue
            
            row, col = idx // 2, idx % 2
            
            # 엔트로피 분포
            entropies = [selection['selection_entropy'] for selection in data['selections']]
            
            axes[row, col].hist(entropies, bins=20, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{stage_name} - Selection Entropy Distribution')
            axes[row, col].set_xlabel('Entropy Value')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].axvline(np.mean(entropies), color='red', linestyle='--', label=f'Mean: {np.mean(entropies):.3f}')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'selection_entropy_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 엔트로피 통계 저장
        entropy_stats = {}
        for stage_name, data in self.tokenselect_data.items():
            if data['selections']:
                entropies = [selection['selection_entropy'] for selection in data['selections']]
                entropy_stats[stage_name] = {
                    'mean_entropy': np.mean(entropies),
                    'std_entropy': np.std(entropies),
                    'min_entropy': np.min(entropies),
                    'max_entropy': np.max(entropies),
                    'total_samples': len(entropies)
                }
        
        with open(os.path.join(output_dir, 'selection_entropy_stats.json'), 'w') as f:
            json.dump(entropy_stats, f, indent=4)
        
        print(f"선택 엔트로피 분석 결과 저장: {output_dir}/selection_entropy_analysis.png")
        return entropy_stats
    
    def analyze_modality_contribution(self, output_dir: str):
        """모달리티 기여도 분석"""
        print("모달리티 기여도 분석 중...")
        
        # 전체 선택 횟수 집계
        total_selections = defaultdict(int)
        for stage_selections in self.modality_selection_counts.values():
            for modality, count in stage_selections.items():
                total_selections[modality] += count
        
        if not total_selections:
            print("선택 데이터가 없습니다.")
            return
        
        # 모달리티별 기여도 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 전체 기여도
        modalities = list(total_selections.keys())
        counts = list(total_selections.values())
        
        ax1.bar(modalities, counts, alpha=0.7)
        ax1.set_title('Overall Modality Contribution')
        ax1.set_xlabel('Modality Index')
        ax1.set_ylabel('Total Selection Count')
        ax1.grid(True, alpha=0.3)
        
        # 스테이지별 기여도 히트맵
        if len(self.modality_selection_counts) > 1:
            stages = list(self.modality_selection_counts.keys())
            all_modalities = set()
            for stage_selections in self.modality_selection_counts.values():
                all_modalities.update(stage_selections.keys())
            
            all_modalities = sorted(list(all_modalities))
            heatmap_data = []
            
            for stage in stages:
                stage_row = []
                for modality in all_modalities:
                    count = self.modality_selection_counts[stage].get(modality, 0)
                    stage_row.append(count)
                heatmap_data.append(stage_row)
            
            sns.heatmap(heatmap_data, 
                       xticklabels=all_modalities, 
                       yticklabels=stages,
                       annot=True, 
                       fmt='d',
                       ax=ax2)
            ax2.set_title('Modality Selection by Stage')
            ax2.set_xlabel('Modality Index')
            ax2.set_ylabel('Stage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'modality_contribution_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 기여도 통계 저장
        contribution_stats = {
            'total_selections': dict(total_selections),
            'stage_breakdown': dict(self.modality_selection_counts),
            'most_contributing_modality': max(total_selections, key=total_selections.get),
            'least_contributing_modality': min(total_selections, key=total_selections.get)
        }
        
        with open(os.path.join(output_dir, 'modality_contribution_stats.json'), 'w') as f:
            json.dump(contribution_stats, f, indent=4)
        
        print(f"모달리티 기여도 분석 결과 저장: {output_dir}/modality_contribution_analysis.png")
        return contribution_stats
    
    def generate_summary_report(self, output_dir: str):
        """종합 분석 보고서 생성"""
        print("종합 분석 보고서 생성 중...")
        
        report = {
            'analysis_summary': {
                'total_stages': len(self.tokenselect_data),
                'stages_analyzed': list(self.tokenselect_data.keys()),
                'total_modality_selections': sum(sum(stage.values()) for stage in self.modality_selection_counts.values())
            },
            'stage_details': {}
        }
        
        for stage_name, data in self.tokenselect_data.items():
            stage_report = {
                'total_selections': len(data['selections']) if data['selections'] else 0,
                'total_scores': len(data['scores']) if data['scores'] else 0
            }
            
            if data['selections']:
                all_selections = []
                entropies = []
                for selection in data['selections']:
                    all_selections.extend(selection['selected_indices'])
                    entropies.append(selection['selection_entropy'])
                
                if all_selections:
                    unique, counts = np.unique(all_selections, return_counts=True)
                    stage_report['selection_analysis'] = {
                        'unique_modalities': len(unique),
                        'most_selected': int(unique[np.argmax(counts)]),
                        'avg_entropy': np.mean(entropies),
                        'selection_distribution': dict(zip(unique.tolist(), counts.tolist()))
                    }
            
            report['stage_details'][stage_name] = stage_report
        
        with open(os.path.join(output_dir, 'tokenselect_analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"종합 분석 보고서 저장: {output_dir}/tokenselect_analysis_report.json")
        return report

def main():
    parser = argparse.ArgumentParser(description='TokenSelect Modality Selection Analysis')
    parser.add_argument('--config', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_heuristicalign_cmnextpsp_b2_rcnn_multiscale_v2_with_visualization/local-sejong2504_heuristicalign_cmnextpsp_rcnn_lr0.01_ep50_v2_with_visualization.py',
                       help='Model config file path')
    parser.add_argument('--checkpoint', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_heuristicalign_cmnextpsp_b2_rcnn_multiscale_v2_with_visualization/epoch_15.pth',
                       help='Model checkpoint file path')
    parser.add_argument('--output-dir', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/tokenselect_analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TokenSelect 분석기 초기화
    analyzer = TokenSelectAnalyzer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    print("=== TokenSelect 모달리티 선택 분석 시작 ===")
    
    # 1. TokenSelect 분석 실행
    print("1. TokenSelect 분석 실행...")
    tokenselect_data = analyzer.run_analysis(num_samples=args.num_samples)
    
    # 2. 모달리티 선택 패턴 분석
    print("2. 모달리티 선택 패턴 분석...")
    selection_stats = analyzer.analyze_modality_selection_patterns(args.output_dir)
    
    # 3. 선택 점수 분석
    print("3. 선택 점수 분석...")
    score_stats = analyzer.analyze_selection_scores(args.output_dir)
    
    # 4. 선택 엔트로피 분석
    print("4. 선택 엔트로피 분석...")
    entropy_stats = analyzer.analyze_selection_entropy(args.output_dir)
    
    # 5. 모달리티 기여도 분석
    print("5. 모달리티 기여도 분석...")
    contribution_stats = analyzer.analyze_modality_contribution(args.output_dir)
    
    # 6. 종합 보고서 생성
    print("6. 종합 분석 보고서 생성...")
    report = analyzer.generate_summary_report(args.output_dir)
    
    print(f"\n=== TokenSelect 분석 완료 ===")
    print(f"결과 디렉토리: {args.output_dir}")
    print(f"분석된 스테이지: {len(tokenselect_data)}개")
    
    # 주요 결과 요약
    if report['stage_details']:
        print("\n=== 주요 결과 요약 ===")
        for stage_name, details in report['stage_details'].items():
            print(f"\n{stage_name}:")
            print(f"  - 총 선택 횟수: {details['total_selections']}")
            
            if 'selection_analysis' in details:
                sel_analysis = details['selection_analysis']
                print(f"  - 고유 모달리티 수: {sel_analysis['unique_modalities']}")
                print(f"  - 가장 많이 선택된 모달리티: {sel_analysis['most_selected']}")
                print(f"  - 평균 엔트로피: {sel_analysis['avg_entropy']:.4f}")
                
                if 'selection_distribution' in sel_analysis:
                    print(f"  - 선택 분포: {sel_analysis['selection_distribution']}")

if __name__ == '__main__':
    main()
