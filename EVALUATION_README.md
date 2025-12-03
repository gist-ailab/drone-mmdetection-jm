# 종합적인 모델 평가 도구

이 도구는 학습된 CMNeXtPSP 모델을 종합적으로 평가하고 다음과 같은 결과를 생성합니다:

- **Inference 이미지**: 모델의 예측 결과 시각화
- **Ground Truth 시각화**: 실제 레이블과 예측 결과 비교
- **CMNeXt Hook 정보**: Feature map, Attention map, 모달리티 기여도 등
- **평가 통계**: mAP, 객체 수 통계 등

## 파일 구조

```
├── comprehensive_evaluation.py      # 메인 평가 스크립트
├── run_comprehensive_evaluation.sh  # 배치 실행 스크립트
└── EVALUATION_README.md            # 이 파일
```

## 사용법

### 1. 단일 모델 평가

```bash
python comprehensive_evaluation.py \
    --config /path/to/config.py \
    --checkpoint /path/to/checkpoint.pth \
    --output-dir ./evaluation_results \
    --num-samples 50 \
    --device cuda:0
```

### 2. 배치 스크립트 사용

```bash
# 모든 모델 전체 평가
./run_comprehensive_evaluation.sh

# 빠른 평가 (시각화 없이)
./run_comprehensive_evaluation.sh --quick

# 특정 모델만 평가
./run_comprehensive_evaluation.sh --model lecun_sejong2504_heuristicalign_10p_cmnextpsp_rcnn_lr0.01_ep50_v2_with_visualization

# 사용 가능한 모델 목록
./run_comprehensive_evaluation.sh --list

# 도움말
./run_comprehensive_evaluation.sh --help
```

## 출력 구조

평가 완료 후 다음과 같은 디렉토리 구조가 생성됩니다:

```
evaluation_results/
├── inference_images/           # 예측 결과 이미지들
│   ├── image001_original.jpg      # 원본 이미지
│   ├── image001_prediction.jpg    # 예측 결과
│   └── image001_comparison.jpg    # GT vs 예측 비교
├── gt_visualizations/         # Ground Truth 시각화
│   └── image001_gt.jpg           # GT 바운딩 박스
├── hook_visualizations/       # CMNeXt Hook 시각화
│   ├── image001_stage1_attention_map.png     # Attention map
│   ├── image001_stage1_rgb_feature_feature_map.png  # Feature map
│   └── image001_hook_info.json              # Hook 정보 JSON
└── evaluation_results/        # 평가 결과
    ├── evaluation_results.json    # 전체 결과
    └── evaluation_summary.json    # 요약 통계
```

## 주요 기능

### 1. 이미지 시각화
- **원본 이미지**: 입력 이미지 저장
- **예측 결과**: 바운딩 박스와 클래스 레이블, 신뢰도 점수
- **Ground Truth**: 실제 레이블 시각화
- **비교 이미지**: GT와 예측 결과를 나란히 배치

### 2. CMNeXt Hook 정보
- **Attention Map**: 각 모달리티의 기여도를 RGB 채널에 매핑
- **Feature Map**: 각 스테이지의 feature map을 그리드로 시각화
- **통계 정보**: Feature의 평균, 표준편차, 최소/최대값
- **모달리티 기여도**: 각 모달리티별 평균 attention weight

### 3. 평가 통계
- **객체 수 통계**: GT vs 예측 객체 수
- **클래스별 분포**: 각 클래스별 검출 결과
- **이미지별 통계**: 이미지당 평균 객체 수

## 설정 옵션

### comprehensive_evaluation.py 옵션

- `--config`: 모델 설정 파일 경로 (필수)
- `--checkpoint`: 체크포인트 파일 경로 (필수)  
- `--output-dir`: 결과 저장 디렉토리 (필수)
- `--num-samples`: 평가할 샘플 수 (-1: 전체, 기본값: -1)
- `--device`: 사용할 디바이스 (기본값: cuda:0)
- `--no-visualization`: 시각화 이미지 저장 생략

### run_comprehensive_evaluation.sh 옵션

- `--quick`: 빠른 평가 (시각화 없이, 20개 샘플만)
- `--model <name>`: 특정 모델만 평가
- `--list`: 사용 가능한 모델 목록 출력
- `--help`: 도움말 출력

## 예제 사용 시나리오

### 시나리오 1: 새로 학습한 모델 평가

```bash
# 1. 특정 모델의 성능 확인
python comprehensive_evaluation.py \
    --config work_dirs/my_model/config.py \
    --checkpoint work_dirs/my_model/best_coco_bbox_mAP_epoch_30.pth \
    --output-dir evaluation_outputs/my_model_eval \
    --num-samples 100

# 2. 결과 확인
ls evaluation_outputs/my_model_eval/
```

### 시나리오 2: 여러 모델 성능 비교

```bash
# 1. 모든 모델 빠른 평가
./run_comprehensive_evaluation.sh --quick

# 2. 결과 비교
ls evaluation_outputs/
```

### 시나리오 3: 특정 이미지에 대한 상세 분석

```bash
# 1. 소수 샘플로 상세 분석
python comprehensive_evaluation.py \
    --config work_dirs/my_model/config.py \
    --checkpoint work_dirs/my_model/best_coco_bbox_mAP_epoch_30.pth \
    --output-dir detailed_analysis \
    --num-samples 10

# 2. Hook 시각화 결과 확인
ls detailed_analysis/hook_visualizations/
```

## 결과 해석

### 1. Hook 정보 JSON 구조

```json
{
  "stage1": {
    "attention_contributions": {
      "modal_0": 0.45,  // RGB 기여도
      "modal_1": 0.35,  // Depth 기여도  
      "modal_2": 0.20   // Event 기여도
    },
    "rgb_feature_stats": {
      "shape": [1, 256, 48, 64],
      "mean": 0.123,
      "std": 0.456,
      "min": -1.234,
      "max": 2.345
    }
  }
}
```

### 2. 평가 통계 JSON 구조

```json
{
  "total_samples": 100,
  "processed_samples": 98,
  "total_gt_objects": 450,
  "total_pred_objects": 423,
  "avg_gt_per_image": 4.59,
  "avg_pred_per_image": 4.32,
  "class_names": ["Enemy", "LandingMarker", "Obstacle", ...]
}
```

## 문제 해결

### 1. CUDA 메모리 부족
```bash
# 배치 크기를 줄이거나 샘플 수를 줄입니다
python comprehensive_evaluation.py --num-samples 20 --device cuda:0
```

### 2. Hook 정보가 저장되지 않음
- CMNeXt 모델이 올바르게 로드되었는지 확인
- Config 파일에 `custom_imports`가 포함되어 있는지 확인

### 3. 시각화 이미지가 생성되지 않음
- OpenCV와 matplotlib이 설치되어 있는지 확인
- 이미지 파일 경로가 올바른지 확인

## 의존성

```bash
pip install torch torchvision
pip install mmengine mmdet
pip install opencv-python matplotlib
pip install numpy pillow
```

## 추가 기능 요청

더 많은 기능이 필요하시면 다음을 고려해보세요:

1. **mAP 계산**: COCO 평가 메트릭 추가
2. **클래스별 분석**: 클래스별 성능 상세 분석
3. **오류 분석**: False Positive/Negative 분석
4. **속도 측정**: 추론 시간 측정
5. **배치 처리**: 더 효율적인 배치 처리

이러한 기능이 필요하시면 언제든 요청해주세요!
