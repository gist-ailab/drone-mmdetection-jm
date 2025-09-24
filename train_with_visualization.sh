#!/bin/bash

# CMNeXt/CMNeXtP 모델 훈련 스크립트 (시각화 Hook 포함)
# 사용법: ./train_with_visualization.sh [cmnext|cmnextp]

MODEL_TYPE=${1:-cmnextp}  # 기본값: cmnextp

echo "🚀 Starting training with CMNeXtVisualizationHook..."
echo "📊 Model type: $MODEL_TYPE"

# GPU 설정
export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 모델 타입에 따른 설정 파일 선택
if [ "$MODEL_TYPE" = "cmnext" ]; then
    CONFIG_FILE="custom_configs/Sejong/a100/a100-sejong2504_heuristicalign_cmnext_rcnn_lr0.01_ep50_v2_with_visualization.py"
    echo "🔧 Using CMNeXt model configuration"
elif [ "$MODEL_TYPE" = "cmnextp" ]; then
    CONFIG_FILE="custom_configs/Sejong/a100/a100-sejong2504_heuristicalign_cmnextpsp_rcnn_lr0.01_ep50_v2_with_visualization.py"
    echo "🔧 Using CMNeXtP model configuration"
else
    echo "❌ Error: Invalid model type. Use 'cmnext' or 'cmnextp'"
    echo "Usage: ./train_with_visualization.sh [cmnext|cmnextp]"
    exit 1
fi

# 시각화 출력 디렉토리 생성
mkdir -p ./visualization_outputs

echo "📁 Visualization outputs will be saved to: ./visualization_outputs"
echo "📈 Wandb logging enabled for visualization"
echo "🔍 Hook will log modality selection statistics and feature maps"

# 훈련 실행
torchrun --nproc_per_node=5 \
    --master_port=29600 \
    tools/train_debug.py \
    --config $CONFIG_FILE \
    --launcher pytorch

echo "✅ Training completed!"
echo "📊 Check Wandb dashboard for visualization results"
echo "📁 Check ./visualization_outputs/ for saved images"

