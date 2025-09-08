import json
import argparse
import numpy as np
from pycocotools.coco import COCO
from sklearn.cluster import KMeans

def analyze_bboxes(ann_file: str, num_clusters: int):
    """
    COCO 형식의 annotation 파일을 분석하여 BBox의 종횡비(aspect ratio)를
    K-means 클러스터링하고, 최적의 앵커 비율을 추천합니다.

    Args:
        ann_file (str): COCO annotation 파일 경로 (e.g., 'labels/train.json')
        num_clusters (int): 추천받을 앵커(클러스터)의 개수
    """
    print(f"Loading annotations from: {ann_file}")
    coco = COCO(ann_file)
    
    aspect_ratios = []
    print("Analyzing bounding box aspect ratios...")
    
    for ann_id in coco.getAnnIds():
        ann = coco.loadAnns(ann_id)[0]
        bbox = ann['bbox']
        # bbox 형식: [x, y, width, height]
        width = bbox[2]
        height = bbox[3]
        
        # 너비 또는 높이가 0인 박스는 분석에서 제외
        if width > 0 and height > 0:
            aspect_ratios.append(width / height)
            
    if not aspect_ratios:
        print("No valid bounding boxes found for analysis.")
        return

    # K-means 클러스터링을 위해 데이터를 2D 배열로 변환
    X = np.array(aspect_ratios).reshape(-1, 1)
    
    print(f"Performing K-means clustering for {num_clusters} anchor ratios...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(X)
    
    # 클러스터의 중심점(추천 앵커 비율)을 추출하고 정렬
    suggested_ratios = sorted(kmeans.cluster_centers_.flatten())
    
    print("\n" + "="*50)
    print("            Analysis Complete")
    print("="*50)
    print(f"Total valid bounding boxes analyzed: {len(aspect_ratios)}")
    print(f"Suggested anchor ratios for {num_clusters} anchors:")
    
    # 소수점 셋째 자리까지 보기 좋게 출력
    formatted_ratios = [f"{ratio:.3f}" for ratio in suggested_ratios]
    print(formatted_ratios)
    print("="*50)
    print("\n💡 이 값을 mmdetection config의 `anchor_generator.ratios`에 적용하세요.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COCO BBox Aspect Ratio Analyzer')
    parser.add_argument(
        '--ann-file', 
        type=str,
        required=True,
        help='Path to the COCO format annotation file (e.g., your_dataset/labels/train.json)'
    )
    parser.add_argument(
        '--num-anchors',
        type=int,
        default=5,
        help='The number of anchor ratios to suggest'
    )
    args = parser.parse_args()
    
    analyze_bboxes(args.ann_file, args.num_anchors)