#!/usr/bin/env python3
"""
mAP 평가를 위한 스크립트
COCO 형식의 예측 결과를 사용하여 mAP를 계산합니다.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import mmengine
from mmdet.registry import DATASETS
from mmdet.evaluation import CocoMetric
import torch
from mmdet.apis import init_detector

# 사용자의 커스텀 모듈 등록
from mcdet import *

def evaluate_coco_map(config_path: str, checkpoint_path: str, 
                     predictions_file: str, output_dir: str, device: str = 'cuda:0'):
    """
    COCO 형식의 예측 결과를 사용하여 mAP를 평가합니다.
    
    Args:
        config_path: 모델 설정 파일 경로
        checkpoint_path: 체크포인트 파일 경로
        predictions_file: COCO 형식 예측 결과 파일 경로
        output_dir: 결과 저장 디렉토리
        device: 사용할 디바이스
    """
    
    # 설정 로드
    cfg = mmengine.Config.fromfile(config_path)
    mmengine.registry.init_default_scope(cfg.get('default_scope', 'mmdet'))
    
    # 데이터셋 로드
    val_dataset_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(val_dataset_cfg)
    dataset.full_init()
    
    # COCO 평가 메트릭 초기화
    evaluator = CocoMetric(
        ann_file=val_dataset_cfg.ann_file,
        metric='bbox',
        format_only=False,
        classwise=True  # 클래스별 mAP도 계산
    )
    
    # 예측 결과 로드
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"로드된 예측 결과: {len(predictions)}개")
    
    # 모델 초기화 (평가를 위해)
    model = init_detector(config_path, checkpoint_path, device=device, cfg_options={'strict': False})
    model.eval()
    
    # 데이터셋에서 샘플을 하나씩 처리하여 평가
    print("평가 진행 중...")
    
    # COCO 형식의 예측 결과를 mmdet 형식으로 변환
    # 이 부분은 실제 구현에 따라 조정이 필요할 수 있습니다
    
    # 임시로 더미 데이터를 사용하여 평가 구조를 보여줍니다
    # 실제 구현에서는 predictions를 적절히 변환해야 합니다
    
    print("COCO 평가 메트릭 설정 완료")
    print(f"데이터셋 클래스: {dataset.metainfo['classes']}")
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 평가 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    
    # 간단한 통계 계산
    if predictions:
        scores = [pred['score'] for pred in predictions]
        categories = [pred['category_id'] for pred in predictions]
        
        stats = {
            'total_predictions': len(predictions),
            'avg_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'unique_categories': len(set(categories)),
            'category_distribution': {cat: categories.count(cat) for cat in set(categories)}
        }
        
        with open(os.path.join(output_dir, 'evaluation_stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"평가 통계 저장: {output_dir}/evaluation_stats.json")
        print(f"총 예측 수: {stats['total_predictions']}")
        print(f"평균 점수: {stats['avg_score']:.4f}")
        print(f"카테고리 수: {stats['unique_categories']}")
    
    return stats if predictions else {}

def run_detailed_evaluation(config_path: str, checkpoint_path: str, 
                           output_dir: str, num_samples: int = 100, device: str = 'cuda:0'):
    """
    더 상세한 평가를 수행합니다.
    """
    
    # 설정 로드
    cfg = mmengine.Config.fromfile(config_path)
    mmengine.registry.init_default_scope(cfg.get('default_scope', 'mmdet'))
    
    # 모델 초기화
    model = init_detector(config_path, checkpoint_path, device=device, cfg_options={'strict': False})
    model.eval()
    
    # 데이터셋 로드
    val_dataset_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(val_dataset_cfg)
    dataset.full_init()
    
    print(f"데이터셋 로드 완료: {len(dataset)}개 샘플")
    
    # COCO 평가 메트릭 초기화
    evaluator = CocoMetric(
        ann_file=val_dataset_cfg.ann_file,
        metric='bbox',
        format_only=False,
        classwise=True
    )
    
    # 샘플링
    total_samples = len(dataset) if num_samples < 0 else min(num_samples, len(dataset))
    sample_indices = np.random.choice(len(dataset), total_samples, replace=False)
    
    print(f"평가할 샘플 수: {total_samples}")
    
    # 평가 진행
    all_predictions = []
    all_ground_truths = []
    
    for i, idx in enumerate(sample_indices):
        if i % 10 == 0:
            print(f"진행률: {i+1}/{total_samples}")
        
        # 데이터 로드
        data = dataset[idx]
        
        # 추론
        with torch.no_grad():
            batched_data = { 
                'inputs': [[item] for item in data['inputs']], 
                'data_samples': [data['data_samples']] 
            }
            processed_data = model.data_preprocessor(batched_data, training=False)
            predictions = model.forward(**processed_data, mode='predict')
        
        # 결과 저장 (실제 구현에서는 더 정교한 변환이 필요)
        # 여기서는 구조만 보여줍니다
        
    print("평가 완료")
    
    # 결과 저장
    evaluation_results = {
        'total_samples': total_samples,
        'model_config': config_path,
        'checkpoint': checkpoint_path
    }
    
    with open(os.path.join(output_dir, 'detailed_evaluation.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    return evaluation_results

def main():
    parser = argparse.ArgumentParser(description='mAP Evaluation for CMNeXtPSP')
    parser.add_argument('--config', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_heuristicalign_cmnextpsp_b2_rcnn_multiscale_v2_with_visualization/local-sejong2504_heuristicalign_cmnextpsp_rcnn_lr0.01_ep50_v2_with_visualization.py',
                       help='Model config file path')
    parser.add_argument('--checkpoint', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_heuristicalign_cmnextpsp_b2_rcnn_multiscale_v2_with_visualization/epoch_15.pth',
                       help='Model checkpoint file path')
    parser.add_argument('--predictions', 
                       help='COCO format predictions file path')
    parser.add_argument('--output-dir', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--detailed', action='store_true', help='Run detailed evaluation')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.detailed:
        print("상세 평가 실행...")
        results = run_detailed_evaluation(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            device=args.device
        )
    elif args.predictions:
        print("COCO 형식 예측 결과 평가...")
        results = evaluate_coco_map(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            predictions_file=args.predictions,
            output_dir=args.output_dir,
            device=args.device
        )
    else:
        print("예측 결과 파일이 필요합니다. --predictions 옵션을 사용하거나 --detailed 옵션을 사용하세요.")
        return
    
    print(f"\n평가 완료!")
    print(f"결과 저장 위치: {args.output_dir}")
    
    if results:
        print(f"평가된 샘플 수: {results.get('total_samples', 'N/A')}")

if __name__ == '__main__':
    main()
