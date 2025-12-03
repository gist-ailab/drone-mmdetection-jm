#!/bin/bash

# 종합적인 모델 평가 실행 스크립트

# 기본 설정
WORK_DIR="/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-mmdetection-jm"
OUTPUT_BASE_DIR="${WORK_DIR}/evaluation_outputs"

# 평가할 모델들 설정
declare -A MODELS=(
    ["lecun_sejong2504_heuristicalign_10p_cmnextpsp_rcnn_lr0.01_ep50_v2_with_visualization"]="${WORK_DIR}/work_dirs/lecun_sejong2504_heuristicalign_10p_cmnextpsp_rcnn_lr0.01_ep50_v2_with_visualization"
    ["lecun-sejong2504_heuristicalign_10p_cmnextpsp_rcnn_lr0.01_ep50_v2_with_visualization"]="${WORK_DIR}/work_dirs/lecun-sejong2504_heuristicalign_10p_cmnextpsp_rcnn_lr0.01_ep50_v2_with_visualization"
)

# 함수: 모델 평가 실행
evaluate_model() {
    local model_name=$1
    local model_dir=$2
    
    echo "=========================================="
    echo "모델 평가 시작: $model_name"
    echo "=========================================="
    
    # 설정 파일과 체크포인트 찾기
    config_file=""
    checkpoint_file=""
    
    # 설정 파일 찾기 (.py 파일)
    if [ -f "$model_dir/${model_name}.py" ]; then
        config_file="$model_dir/${model_name}.py"
    else
        # 디렉토리에서 .py 파일 찾기
        config_file=$(find "$model_dir" -name "*.py" -type f | head -1)
    fi
    
    # 체크포인트 파일 찾기 (best 또는 최신 epoch)
    if [ -f "$model_dir/best_coco_bbox_mAP_epoch_*.pth" ]; then
        checkpoint_file=$(ls -t "$model_dir"/best_coco_bbox_mAP_epoch_*.pth | head -1)
    elif [ -f "$model_dir/epoch_*.pth" ]; then
        checkpoint_file=$(ls -t "$model_dir"/epoch_*.pth | head -1)
    fi
    
    # 파일 존재 확인
    if [ ! -f "$config_file" ]; then
        echo "Error: 설정 파일을 찾을 수 없습니다: $config_file"
        return 1
    fi
    
    if [ ! -f "$checkpoint_file" ]; then
        echo "Error: 체크포인트 파일을 찾을 수 없습니다: $checkpoint_file"
        return 1
    fi
    
    echo "설정 파일: $config_file"
    echo "체크포인트: $checkpoint_file"
    
    # 출력 디렉토리 설정
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_dir="${OUTPUT_BASE_DIR}/${model_name}_${timestamp}"
    
    echo "출력 디렉토리: $output_dir"
    
    # 평가 실행
    cd "$WORK_DIR"
    python comprehensive_evaluation.py \
        --config "$config_file" \
        --checkpoint "$checkpoint_file" \
        --output-dir "$output_dir" \
        --num-samples 50 \
        --device cuda:0
    
    if [ $? -eq 0 ]; then
        echo "✅ 모델 평가 완료: $model_name"
        echo "결과 저장 위치: $output_dir"
    else
        echo "❌ 모델 평가 실패: $model_name"
        return 1
    fi
    
    echo ""
}

# 함수: 빠른 평가 (시각화 없이)
evaluate_model_quick() {
    local model_name=$1
    local model_dir=$2
    export CUDA_VISIBLE_DEVICES=2,3
 
    echo "=========================================="
    echo "빠른 모델 평가 시작: $model_name"
    echo "=========================================="
    
    # 설정 파일과 체크포인트 찾기 (위와 동일)
    config_file=""
    checkpoint_file=""
    
    if [ -f "$model_dir/${model_name}.py" ]; then
        config_file="$model_dir/${model_name}.py"
    else
        config_file=$(find "$model_dir" -name "*.py" -type f | head -1)
    fi
    
    if [ -f "$model_dir/best_coco_bbox_mAP_epoch_*.pth" ]; then
        checkpoint_file=$(ls -t "$model_dir"/best_coco_bbox_mAP_epoch_*.pth | head -1)
    elif [ -f "$model_dir/epoch_*.pth" ]; then
        checkpoint_file=$(ls -t "$model_dir"/epoch_*.pth | head -1)
    fi
    
    if [ ! -f "$config_file" ] || [ ! -f "$checkpoint_file" ]; then
        echo "Error: 필요한 파일을 찾을 수 없습니다"
        return 1
    fi
    
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_dir="${OUTPUT_BASE_DIR}/${model_name}_quick_${timestamp}"
    
    cd "$WORK_DIR"
    python comprehensive_evaluation.py \
        --config "$config_file" \
        --checkpoint "$checkpoint_file" \
        --output-dir "$output_dir" \
        --num-samples 20 \
        --device cuda:0 \
        --no-visualization
    
    if [ $? -eq 0 ]; then
        echo "✅ 빠른 평가 완료: $model_name"
    else
        echo "❌ 빠른 평가 실패: $model_name"
        return 1
    fi
}

# 메인 실행 부분
echo "종합적인 모델 평가 스크립트"
echo "=============================="

# 출력 베이스 디렉토리 생성
mkdir -p "$OUTPUT_BASE_DIR"

# 사용법 출력
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "사용법:"
    echo "  $0                    # 모든 모델 전체 평가"
    echo "  $0 --quick           # 모든 모델 빠른 평가"
    echo "  $0 --model <name>    # 특정 모델만 평가"
    echo "  $0 --list           # 사용 가능한 모델 목록"
    echo ""
    echo "옵션:"
    echo "  --quick             시각화 없이 빠른 평가"
    echo "  --model <name>      특정 모델만 평가"
    echo "  --list              사용 가능한 모델 목록 출력"
    exit 0
fi

# 모델 목록 출력
if [ "$1" = "--list" ]; then
    echo "사용 가능한 모델들:"
    for model_name in "${!MODELS[@]}"; do
        echo "  - $model_name"
    done
    exit 0
fi

# 특정 모델만 평가
if [ "$1" = "--model" ] && [ -n "$2" ]; then
    model_name="$2"
    if [[ -n "${MODELS[$model_name]}" ]]; then
        if [ "$3" = "--quick" ]; then
            evaluate_model_quick "$model_name" "${MODELS[$model_name]}"
        else
            evaluate_model "$model_name" "${MODELS[$model_name]}"
        fi
    else
        echo "Error: 모델 '$model_name'을 찾을 수 없습니다."
        echo "사용 가능한 모델: ${!MODELS[@]}"
        exit 1
    fi
    exit 0
fi

# 모든 모델 평가
echo "모든 모델 평가를 시작합니다..."
echo "총 ${#MODELS[@]}개 모델"
echo ""

success_count=0
total_count=${#MODELS[@]}

for model_name in "${!MODELS[@]}"; do
    model_dir="${MODELS[$model_name]}"
    
    if [ ! -d "$model_dir" ]; then
        echo "Warning: 모델 디렉토리가 존재하지 않습니다: $model_dir"
        continue
    fi
    
    if [ "$1" = "--quick" ]; then
        evaluate_model_quick "$model_name" "$model_dir"
    else
        evaluate_model "$model_name" "$model_dir"
    fi
    
    if [ $? -eq 0 ]; then
        ((success_count++))
    fi
done

echo "=========================================="
echo "전체 평가 완료"
echo "성공: $success_count/$total_count"
echo "결과 저장 위치: $OUTPUT_BASE_DIR"
echo "=========================================="

# 결과 요약 생성
summary_file="${OUTPUT_BASE_DIR}/evaluation_summary_$(date +"%Y%m%d_%H%M%S").txt"
{
    echo "모델 평가 요약"
    echo "============="
    echo "평가 시간: $(date)"
    echo "성공한 모델: $success_count/$total_count"
    echo ""
    echo "평가된 모델들:"
    for model_name in "${!MODELS[@]}"; do
        echo "  - $model_name"
    done
} > "$summary_file"

echo "요약 파일 생성: $summary_file"
