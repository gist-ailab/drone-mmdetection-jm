# example_cmnext_hook_usage.py
"""
CMNeXtVisualizationHook 사용 예시

이 파일은 CMNeXtVisualizationHook을 사용하여 CMNeXt/CMNeXtP 모델의
모달리티 선택 및 융합 과정을 시각화하는 방법을 보여줍니다.
"""

from mmengine import Config
from mmdet.registry import RUNNERS
from mmengine.runner import Runner

def create_config_with_cmnext_hook():
    """CMNeXtVisualizationHook이 포함된 설정을 생성합니다."""
    
    # 기본 설정
    cfg = Config()
    
    # 모델 설정 (예시)
    cfg.model = Config()
    cfg.model.type = 'FasterRCNN'
    cfg.model.backbone = Config()
    cfg.model.backbone.type = 'CMNeXtPBackbone'  # 또는 'CMNextBackbone'
    cfg.model.backbone.backbone = 'CMNeXtP-B2'
    cfg.model.backbone.modals = ['rgb', 'depth', 'event', 'lidar']
    cfg.model.backbone.out_indices = (0, 1, 2, 3)
    
    # Hook 설정
    cfg.default_hooks = Config()
    cfg.default_hooks.logger = Config()
    cfg.default_hooks.logger.type = 'TextLoggerHook'
    
    # CMNeXtVisualizationHook 추가
    cfg.custom_hooks = [
        Config(
            type='CMNeXtVisualizationHook',
            log_interval=1000,  # 1000 iteration마다 로깅
            log_training=False,  # 훈련 중 로깅 비활성화
            log_validation=True,  # 검증 중 로깅 활성화
            save_images=True,  # 이미지 저장 활성화
            image_save_dir='./visualization_outputs'  # 이미지 저장 디렉토리
        )
    ]
    
    # Wandb 설정 (선택사항)
    cfg.vis_backends = [
        Config(type='WandbVisBackend',
               init_kwargs=Config(
                   project='cmnext-visualization',
                   name='cmnext-hook-demo'
               ))
    ]
    
    return cfg

def main():
    """메인 실행 함수"""
    
    # 설정 생성
    cfg = create_config_with_cmnext_hook()
    
    # Runner 생성 및 실행
    runner = Runner.from_cfg(cfg)
    
    # 훈련 시작
    runner.train()
    
    # 또는 검증만 실행
    # runner.val()

if __name__ == '__main__':
    main()

# 사용법:
# 1. 이 파일을 실행하면 CMNeXtVisualizationHook이 활성화됩니다.
# 2. Hook은 검증 중에 각 스테이지별로 다음을 로깅합니다:
#    - 모달리티 점수 통계 (Soft Mix 또는 Hard Selection)
#    - Feature map 시각화 (PPX, FRM input, FFM output)
#    - 모달리티 기여도 맵 (CMNeXtP의 경우)
#    - 채널별 승자 인덱스 맵 (CMNeXt의 경우)
# 3. 모든 시각화는 Wandb에 로깅되며, 로컬에도 저장됩니다.

# 주의사항:
# - Hook은 CMNeXt/CMNeXtP 백본이 있는 모델에서만 작동합니다.
# - Wandb가 설치되어 있지 않으면 텍스트 통계만 로깅됩니다.
# - 시각화는 검증 중에만 수행되며, 각 epoch의 첫 번째 배치에서만 로깅됩니다.

