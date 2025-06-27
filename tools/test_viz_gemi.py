from mmdet.apis import init_detector, inference_detector
import mmcv

cfg = 'custom_configs/DELIVER/hinton-deliver_geminifusion_rcnn_lr0.01.py'
ckpt = 'work_dirs/.../latest.pth'
model = init_detector(cfg, ckpt, device='cuda:0')

# ── 각 모달 경로를 dict 로 주입 (dataset 파이프라인과 동일한 key 사용) ──
inputs = dict(
    rgb   = 'demo/DELIVER/rgb_0001.png',
    depth = 'demo/DELIVER/depth_0001.png',
    event = 'demo/DELIVER/event_0001.png',
    lidar = 'demo/DELIVER/lidar_0001.png'
)

result = inference_detector(model, inputs)

# 내부 visualizer 사용 – out_file 없으면 pop-up 창
model.visualize(
    image=inputs['rgb', 'depth', 'lidar', 'event'],      # 기준 캔버스
    data_sample=result,
    out_file='demo_vis.png')  # 저장 경로
