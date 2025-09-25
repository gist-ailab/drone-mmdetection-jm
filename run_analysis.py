#!/usr/bin/env python3
"""
CMNeXtPSP 모델 분석 실행 스크립트
데이터 시각화, mAP 평가, FRM 분석을 통합적으로 실행합니다.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
import time

def run_command(cmd, description):
    """명령어 실행 및 결과 출력"""
    print(f"\n{'='*50}")
    print(f"실행 중: {description}")
    print(f"명령어: {cmd}")
    print(f"{'='*50}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        end_time = time.time()
        print(f"✅ {description} 완료 (소요시간: {end_time - start_time:.2f}초)")
        if result.stdout:
            print("출력:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 실패")
        print(f"오류: {e}")
        if e.stderr:
            print(f"에러 메시지: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='CMNeXtPSP Model Analysis Runner')
    parser.add_argument('--config', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_heuristicalign_cmnextpsp_b2_rcnn_multiscale_v2_with_visualization/local-sejong2504_heuristicalign_cmnextpsp_rcnn_lr0.01_ep50_v2_with_visualization.py',
                       help='Model config file path')
    parser.add_argument('--checkpoint', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/sejong2504_heuristicalign_cmnextpsp_b2_rcnn_multiscale_v2_with_visualization/best_coco_bbox_mAP_epoch_15.pth',
                       help='Model checkpoint file path')
    parser.add_argument('--output-dir', 
                       default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/analysis_results',
                       help='Base directory to save all analysis results')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of samples to analyze')
    parser.add_argument('--score-threshold', type=float, default=0.5, help='Score threshold for predictions')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip visualization step')
    parser.add_argument('--skip-frm-analysis', action='store_true', help='Skip FRM analysis step')
    parser.add_argument('--skip-map-evaluation', action='store_true', help='Skip mAP evaluation step')
    args = parser.parse_args()
    
    # 결과 디렉토리 생성
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 CMNeXtPSP 모델 분석 시작")
    print(f"설정 파일: {args.config}")
    print(f"체크포인트: {args.checkpoint}")
    print(f"결과 디렉토리: {args.output_dir}")
    print(f"분석 샘플 수: {args.num_samples}")
    
    # 분석 결과 저장
    analysis_results = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'num_samples': args.num_samples,
        'score_threshold': args.score_threshold,
        'device': args.device,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'steps_completed': [],
        'steps_failed': []
    }
    
    # 1. 데이터 시각화 및 예측 저장
    if not args.skip_visualization:
        vis_output_dir = base_output_dir / 'visualization'
        vis_cmd = f"""python analysis_visualization.py \
            --config {args.config} \
            --checkpoint {args.checkpoint} \
            --output-dir {vis_output_dir} \
            --num-samples {args.num_samples} \
            --score-threshold {args.score_threshold} \
            --device {args.device}"""
        
        if run_command(vis_cmd, "데이터 시각화 및 예측 저장"):
            analysis_results['steps_completed'].append('visualization')
            analysis_results['visualization_output'] = str(vis_output_dir)
        else:
            analysis_results['steps_failed'].append('visualization')
    
    # 2. FRM 모듈 선택 분석
    if not args.skip_frm_analysis:
        frm_output_dir = base_output_dir / 'frm_analysis'
        frm_cmd = f"""python frm_analysis.py \
            --config {args.config} \
            --checkpoint {args.checkpoint} \
            --output-dir {frm_output_dir} \
            --num-samples {args.num_samples} \
            --device {args.device}"""
        
        if run_command(frm_cmd, "FRM 모듈 선택 분석"):
            analysis_results['steps_completed'].append('frm_analysis')
            analysis_results['frm_analysis_output'] = str(frm_output_dir)
        else:
            analysis_results['steps_failed'].append('frm_analysis')
    
    # 3. mAP 평가
    if not args.skip_map_evaluation:
        map_output_dir = base_output_dir / 'map_evaluation'
        predictions_file = base_output_dir / 'visualization' / 'coco_predictions.json'
        
        if predictions_file.exists():
            map_cmd = f"""python evaluate_map.py \
                --config {args.config} \
                --checkpoint {args.checkpoint} \
                --predictions {predictions_file} \
                --output-dir {map_output_dir} \
                --num-samples {args.num_samples} \
                --device {args.device}"""
            
            if run_command(map_cmd, "mAP 평가"):
                analysis_results['steps_completed'].append('map_evaluation')
                analysis_results['map_evaluation_output'] = str(map_output_dir)
            else:
                analysis_results['steps_failed'].append('map_evaluation')
        else:
            print(f"⚠️ 예측 결과 파일을 찾을 수 없습니다: {predictions_file}")
            print("시각화 단계를 먼저 실행하거나 --skip-visualization을 제거하세요.")
            analysis_results['steps_failed'].append('map_evaluation')
    
    # 4. 종합 보고서 생성
    print(f"\n{'='*50}")
    print("종합 분석 보고서 생성")
    print(f"{'='*50}")
    
    # 결과 요약
    total_steps = len([x for x in [not args.skip_visualization, not args.skip_frm_analysis, not args.skip_map_evaluation] if x])
    completed_steps = len(analysis_results['steps_completed'])
    failed_steps = len(analysis_results['steps_failed'])
    
    print(f"\n📊 분석 결과 요약:")
    print(f"  - 전체 단계: {total_steps}")
    print(f"  - 완료된 단계: {completed_steps}")
    print(f"  - 실패한 단계: {failed_steps}")
    
    if analysis_results['steps_completed']:
        print(f"\n✅ 완료된 단계:")
        for step in analysis_results['steps_completed']:
            print(f"  - {step}")
    
    if analysis_results['steps_failed']:
        print(f"\n❌ 실패한 단계:")
        for step in analysis_results['steps_failed']:
            print(f"  - {step}")
    
    # 결과 디렉토리 정보
    print(f"\n📁 결과 디렉토리:")
    print(f"  - 기본 디렉토리: {base_output_dir}")
    
    if 'visualization_output' in analysis_results:
        print(f"  - 시각화 결과: {analysis_results['visualization_output']}")
        print(f"    - 이미지: {analysis_results['visualization_output']}/visualizations/")
        print(f"    - 예측 결과: {analysis_results['visualization_output']}/coco_predictions.json")
    
    if 'frm_analysis_output' in analysis_results:
        print(f"  - FRM 분석 결과: {analysis_results['frm_analysis_output']}")
        print(f"    - 분석 이미지: {analysis_results['frm_analysis_output']}/*.png")
        print(f"    - 통계 데이터: {analysis_results['frm_analysis_output']}/*.json")
    
    if 'map_evaluation_output' in analysis_results:
        print(f"  - mAP 평가 결과: {analysis_results['map_evaluation_output']}")
    
    # 분석 결과 저장
    results_file = base_output_dir / 'analysis_summary.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n💾 분석 요약 저장: {results_file}")
    
    if completed_steps == total_steps:
        print(f"\n🎉 모든 분석이 성공적으로 완료되었습니다!")
    elif completed_steps > 0:
        print(f"\n⚠️ 일부 분석이 완료되었습니다. 실패한 단계를 확인하세요.")
    else:
        print(f"\n❌ 모든 분석이 실패했습니다. 설정을 확인하세요.")
    
    return analysis_results

if __name__ == '__main__':
    main()
