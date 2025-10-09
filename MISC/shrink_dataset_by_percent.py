#!/usr/bin/env python3
"""
데이터셋 샘플링 스크립트

COCO 형식의 annotation 파일에서 지정된 비율만큼 데이터를 샘플링하여
새로운 annotation 파일을 생성합니다.

사용법:
    python create_dataset_sample.py --json_path /path/to/annotation.json --ratio 0.1 --postfix _10percent

인자:
    --json_path: 원본 COCO 형식 annotation 파일 경로
    --ratio: 샘플링할 비율 (0.0 ~ 1.0, 예: 0.1은 10%)
    --postfix: 출력 파일명에 추가할 접미사 (예: _10percent)
"""

import json
import argparse
import random
import os
from pathlib import Path
from typing import Dict, List, Any


def load_coco_annotation(json_path: str) -> Dict[str, Any]:
    """COCO 형식 annotation 파일을 로드합니다."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sample_dataset(coco_data: Dict[str, Any], ratio: float, seed: int = 42) -> Dict[str, Any]:
    """
    COCO 데이터셋에서 지정된 비율만큼 샘플링합니다.
    
    Args:
        coco_data: COCO 형식의 데이터
        ratio: 샘플링할 비율 (0.0 ~ 1.0)
        seed: 랜덤 시드
    
    Returns:
        샘플링된 COCO 형식 데이터
    """
    random.seed(seed)
    
    # 원본 데이터 복사
    sampled_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data.get('categories', []),
        'images': [],
        'annotations': []
    }
    
    # 이미지 개수 계산
    total_images = len(coco_data['images'])
    sample_size = int(total_images * ratio)
    
    print(f"원본 이미지 개수: {total_images}")
    print(f"샘플링할 이미지 개수: {sample_size} ({ratio*100:.1f}%)")
    
    # 이미지 ID를 기준으로 랜덤 샘플링
    all_image_ids = [img['id'] for img in coco_data['images']]
    sampled_image_ids = random.sample(all_image_ids, sample_size)
    sampled_image_ids_set = set(sampled_image_ids)
    
    # 샘플링된 이미지들 추가
    for img in coco_data['images']:
        if img['id'] in sampled_image_ids_set:
            sampled_data['images'].append(img)
    
    # 샘플링된 이미지에 해당하는 annotation들 추가
    for ann in coco_data['annotations']:
        if ann['image_id'] in sampled_image_ids_set:
            sampled_data['annotations'].append(ann)
    
    print(f"샘플링된 annotation 개수: {len(sampled_data['annotations'])}")
    
    return sampled_data


def save_sampled_annotation(sampled_data: Dict[str, Any], output_path: str) -> None:
    """샘플링된 데이터를 JSON 파일로 저장합니다."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"샘플링된 annotation 파일이 저장되었습니다: {output_path}")


def create_output_path(json_path: str, postfix: str) -> str:
    """출력 파일 경로를 생성합니다."""
    path = Path(json_path)
    output_filename = f"{path.stem}{postfix}{path.suffix}"
    return str(path.parent / output_filename)


def main():
    parser = argparse.ArgumentParser(
        description='COCO 형식 annotation 파일에서 지정된 비율만큼 데이터를 샘플링합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
    python create_dataset_sample.py --json_path train.json --ratio 0.1 --postfix _10percent
    python create_dataset_sample.py --json_path val.json --ratio 0.2 --postfix _20percent
        """
    )
    
    parser.add_argument(
        '--json_path', 
        type=str, 
        required=True,
        help='원본 COCO 형식 annotation 파일 경로'
    )
    
    parser.add_argument(
        '--ratio', 
        type=float, 
        required=True,
        help='샘플링할 비율 (0.0 ~ 1.0, 예: 0.1은 10퍼센트)'
    )
    
    parser.add_argument(
        '--postfix', 
        type=str, 
        required=True,
        help='출력 파일명에 추가할 접미사 (예: _10percent)'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='랜덤 시드 (기본값: 42)'
    )
    
    args = parser.parse_args()
    
    # 입력 검증
    if not os.path.exists(args.json_path):
        print(f"오류: 파일을 찾을 수 없습니다: {args.json_path}")
        return
    
    if not (0.0 < args.ratio <= 1.0):
        print(f"오류: ratio는 0.0과 1.0 사이의 값이어야 합니다. 입력값: {args.ratio}")
        return
    
    try:
        # 원본 annotation 파일 로드
        print(f"원본 annotation 파일 로드 중: {args.json_path}")
        coco_data = load_coco_annotation(args.json_path)
        
        # 데이터셋 샘플링
        print(f"데이터셋 샘플링 중... (비율: {args.ratio*100:.1f}%)")
        sampled_data = sample_dataset(coco_data, args.ratio, args.seed)
        
        # 출력 파일 경로 생성
        output_path = create_output_path(args.json_path, args.postfix)
        
        # 샘플링된 데이터 저장
        save_sampled_annotation(sampled_data, output_path)
        
        print("샘플링 완료!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return


if __name__ == "__main__":
    main()
