# my_hooks/vis_log_hook.py

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS

@HOOKS.register_module()
class VisLogHook(Hook):
    """
    Validation/Test Epoch이 시작될 때마다 모델의 로깅 플래그를 리셋하는 훅.
    이를 통해 매 Validation 마다 첫 번째 배치 이미지를 한 번씩만 로깅할 수 있습니다.
    """
    def _before_val_epoch(self, runner: Runner) -> None:
        """Validation Epoch 시작 전 호출됩니다."""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model. _has_logged_this_epoch = False
        
    def _before_test_epoch(self, runner: Runner) -> None:
        """Test Epoch 시작 전 호출됩니다."""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model._has_logged_this_epoch = False