
CUDA_VISIBLE_DEVICES=3,4,5,6,7 \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
torchrun --nproc_per_node=5 \
--master_port=29600 \
tools/train_debug.py \
--config custom_configs/DELIVER/a100-sejong2504_cmnextp_rcnn_lr0.01_ep50_v2.py \
--launcher pytorch