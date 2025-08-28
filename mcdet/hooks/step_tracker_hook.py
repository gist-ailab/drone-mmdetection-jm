# mcdet/hooks/step_tracker_hook.py
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.registry import HOOKS

@HOOKS.register_module()
class StepTrackerHook(Hook):
    """
    매 반복(iteration)마다 Runner의 iter 값을 모델의 속성으로 주입하는 훅.
    이를 통해 모델의 forward 함수 내에서 현재 step을 알 수 있습니다.
    """
    def _before_train_iter(self,
                           runner: Runner,
                           batch_idx: int,
                           data_batch: dict = None) -> None:
        """훈련 반복이 시작되기 직전에 호출됩니다."""
        # 현재 iteration(global step) 값을 가져옵니다.
        current_iter = runner.iter

        # 모델에 'iter'라는 이름의 속성으로 현재 스텝 값을 설정합니다.
        # 분산 학습(DDP) 시에는 runner.model이 MMDistributedDataParallel로 감싸져 있으므로,
        # 실제 모델은 .module 속성을 통해 접근해야 합니다.
        if hasattr(runner.model, 'module'):
            runner.model.module.iter = current_iter
        else:
            runner.model.iter = current_iter