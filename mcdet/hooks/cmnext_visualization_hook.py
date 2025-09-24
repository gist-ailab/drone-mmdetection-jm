# mcdet/hooks/cmnext_visualization_hook.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 비-인터랙티브 백엔드 설정
import matplotlib.pyplot as plt
import torch.distributed as dist
from typing import List, Dict, Any, Optional
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS

try:
    import wandb
except ImportError:
    wandb = None


@HOOKS.register_module()
class CMNeXtVisualizationHook(Hook):
    """
    CMNeXt/CMNeXtP 모델의 모달리티 선택 및 융합 과정을 시각화하고 통계를 수집하는 Hook.
    
    이 Hook은 모델의 forward 과정에서 다음을 수행합니다:
    1. 각 스테이지별 모달리티 점수 통계 수집
    2. Soft Mix 기여도 맵 시각화 (CMNeXtP)
    3. Hard Selection 승자 인덱스 맵 시각화 (CMNeXt)
    4. Feature map 시각화
    5. Wandb 로깅
    """
    
    def __init__(self,
                 log_interval: int = 1000,
                 log_training: bool = False,
                 log_validation: bool = True,
                 save_images: bool = True,
                 image_save_dir: str = './visualization_outputs'):
        """
        Args:
            log_interval (int): 로깅 간격 (iteration 단위)
            log_training (bool): 훈련 중 로깅 여부
            log_validation (bool): 검증 중 로깅 여부
            save_images (bool): 이미지 저장 여부
            image_save_dir (str): 이미지 저장 디렉토리
        """
        self.log_interval = log_interval
        self.log_training = log_training
        self.log_validation = log_validation
        self.save_images = save_images
        self.image_save_dir = image_save_dir
        self.iteration_count = 0
        self._has_logged_this_epoch = False
        
        # 이미지 저장 디렉토리 생성
        if self.save_images:
            import os
            os.makedirs(self.image_save_dir, exist_ok=True)
    
    def _before_val_epoch(self, runner: Runner) -> None:
        """Validation Epoch 시작 전 호출됩니다."""
        self._has_logged_this_epoch = False
        
    def _before_test_epoch(self, runner: Runner) -> None:
        """Test Epoch 시작 전 호출됩니다."""
        self._has_logged_this_epoch = False
    
    def _after_train_iter(self, runner: Runner, batch_idx: int, data_batch: Dict[str, Any] = None, outputs: Dict[str, Any] = None) -> None:
        """Training iteration 후 호출됩니다."""
        if not self.log_training:
            return
            
        self.iteration_count += 1
        if self.iteration_count % self.log_interval == 0:
            self._log_modality_statistics(runner, is_training=True)
    
    def _after_val_iter(self, runner: Runner, batch_idx: int, data_batch: Dict[str, Any] = None, outputs: Dict[str, Any] = None) -> None:
        """Validation iteration 후 호출됩니다."""
        if not self.log_validation:
            return
            
        # 각 epoch마다 첫 번째 배치에서만 로깅
        if not self._has_logged_this_epoch and batch_idx == 0:
            self._log_modality_statistics(runner, is_training=False)
            self._has_logged_this_epoch = True
    
    def _log_modality_statistics(self, runner: Runner, is_training: bool = None) -> None:
        """모달리티 통계를 로깅합니다."""
        # Distributed training 체크
        is_dist = dist.is_available() and dist.is_initialized()
        if is_dist and dist.get_rank() != 0:
            return
            
        if wandb is None:
            return
            
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        
        # CMNeXt/CMNeXtP 백본 찾기
        backbone = self._find_cmnext_backbone(model)
        if backbone is None:
            return
            
        # Hook을 모델에 연결
        if hasattr(backbone, 'cmnext_model'):
            cmnext_model = backbone.cmnext_model.backbone
            cmnext_model._visualization_hook = self
    
    def _find_cmnext_backbone(self, model) -> Optional[Any]:
        """모델에서 CMNeXt 백본을 찾습니다."""
        if hasattr(model, 'backbone'):
            return model.backbone
        elif hasattr(model, 'neck') and hasattr(model.neck, 'backbone'):
            return model.neck.backbone
        return None
    
# CMNeXtP와 CMNeXt의 통계 로깅은 모델의 forward 과정에서 직접 호출됨
    
    def log_soft_mix_stats(self, 
                          attention_map: torch.Tensor, 
                          modality_names: List[str], 
                          log_prefix: str, 
                          batch_idx: int = 0) -> None:
        """
        Soft Mix의 기여도 통계를 계산하고 시각화하여 wandb에 로깅합니다.
        CMNeXtP에서 사용됩니다.
        """
        if wandb is None or attention_map is None:
            return

        # 텍스트 통계 로깅 (평균 기여도)
        avg_contributions = attention_map[batch_idx].mean(dim=[1, 2])
        
        stats_dict = {}
        for i, name in enumerate(modality_names):
            stats_dict[f"{log_prefix}_contribution/{name}"] = avg_contributions[i].item()
        wandb.log(stats_dict)

        # 시각화 맵 로깅 (RGB 기여도 맵)
        num_modals = attention_map.shape[1]
        if num_modals >= 3:
            contribution_rgb = attention_map[batch_idx][:3].permute(1, 2, 0)
        elif num_modals == 2:
            zeros = torch.zeros_like(attention_map[batch_idx][0])
            contribution_rgb = torch.stack([attention_map[batch_idx][0], attention_map[batch_idx][1], zeros], dim=-1)
        else:  # num_modals == 1
            ch = attention_map[batch_idx][0]
            contribution_rgb = torch.stack([ch, ch, ch], dim=-1)

        # 0~1 범위를 0~255 범위의 이미지로 변환
        color_map_np = (contribution_rgb * 255).byte().cpu().numpy()
        
        # wandb에 이미지 로깅
        caption = ""
        for i, name in enumerate(modality_names):
            if i < 3:
                caption += f"{['R', 'G', 'B'][i]}:{name} "
                
        wandb.log({f"{log_prefix}_contribution_map": wandb.Image(color_map_np, caption=caption.strip())})
    
    def log_hard_selection_stats(self,
                                x_scores: List[torch.Tensor],
                                winner_indices: torch.Tensor,
                                modality_names: List[str],
                                log_prefix: str,
                                batch_idx: int = 0) -> None:
        """
        Hard Selection의 통계를 로깅합니다.
        CMNeXt에서 사용됩니다.
        """
        if wandb is None or x_scores is None:
            return
            
        # 점수 통계 로깅
        stats_dict = {}
        for i, modal in enumerate(modality_names):
            if i < len(x_scores):
                stats_dict[f"{log_prefix}_score_{modal}"] = x_scores[i].mean().item()
        wandb.log(stats_dict)
        
        # 승자 인덱스 맵 시각화
        if winner_indices is not None:
            winner_map_image = self.create_channel_winners_image(
                winner_indices=winner_indices,
                modality_names=modality_names,
                title=f'{log_prefix} - Modality Winner per Channel',
                batch_idx=batch_idx
            )
            wandb.log({f"{log_prefix}_channel_winners": wandb.Image(winner_map_image)})
    
    def log_feature_maps(self,
                        feature_maps: Dict[str, torch.Tensor],
                        log_prefix: str,
                        batch_idx: int = 0) -> None:
        """
        Feature map들을 시각화하여 로깅합니다.
        """
        if wandb is None:
            return
            
        for name, tensor in feature_maps.items():
            if tensor is not None:
                feature_image = self.create_tensor_grid_image(
                    tensor=tensor,
                    title=f"{log_prefix} - {name}",
                    batch_idx=batch_idx
                )
                wandb.log({f"{log_prefix}_{name}": wandb.Image(feature_image)})
    
    def create_soft_mix_contribution_map(self, 
                                       attention_map: torch.Tensor, 
                                       batch_idx: int = 0) -> np.ndarray:
        """
        Soft-Mix의 결과인 attention_map을 입력받아, 각 모달리티의 기여도를
        RGB 채널에 매핑한 NumPy 이미지 배열을 생성합니다.
        """
        # 배치에서 하나를 선택하고 CPU로 이동
        attention_map_single = attention_map[batch_idx].cpu()
        num_modals, h, w = attention_map_single.shape

        # (H, W, 3) 크기의 빈 RGB 캔버스 생성
        contribution_rgb = torch.zeros((h, w, 3), dtype=torch.float32)

        # 각 모달리티의 기여도를 RGB 채널에 순서대로 매핑
        if num_modals >= 1:
            contribution_rgb[:, :, 0] = attention_map_single[0]  # 첫 번째 모달리티 -> Red 채널
        if num_modals >= 2:
            contribution_rgb[:, :, 1] = attention_map_single[1]  # 두 번째 모달리티 -> Green 채널
        if num_modals >= 3:
            contribution_rgb[:, :, 2] = attention_map_single[2]  # 세 번째 모달리티 -> Blue 채널

        # 0~1 범위의 float 값을 0~255 범위의 uint8 값으로 변환
        image_array = (contribution_rgb.numpy() * 255).astype(np.uint8)

        return image_array

    def create_tensor_grid_image(self,
                                tensor: torch.Tensor,
                                title: str = 'Feature Map Channels',
                                batch_idx: int = 0) -> np.ndarray:
        """
        (B, C, H, W) 또는 (C, H, W) 형태의 텐서를 입력받아, 모든 채널을
        그리드 형태로 시각화하고 그 결과를 NumPy 이미지 배열로 반환합니다.
        """
        if tensor.dim() == 4:
            tensor = tensor[batch_idx]
        
        tensor_np = tensor.detach().cpu().numpy()
        C, H, W = tensor_np.shape

        ncols = int(np.ceil(np.sqrt(C)))
        nrows = int(np.ceil(C / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
        axes = axes.flatten()
        
        fig.suptitle(title, fontsize=16)
        
        for i in range(C):
            channel_feature = tensor_np[i]
            if channel_feature.max() > channel_feature.min():
                channel_feature = (channel_feature - channel_feature.min()) / (channel_feature.max() - channel_feature.min())
                
            ax = axes[i]
            ax.imshow(channel_feature, cmap='jet')
            ax.axis('off')
            ax.set_title(f"Ch: {i}", fontsize=10)
        
        for j in range(C, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Figure를 파일로 저장하는 대신 RGB 버퍼로 렌더링
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Figure 객체를 닫아 메모리 누수 방지
        plt.close(fig)

        return img_array

    def create_channel_winners_image(self,
                                   winner_indices: torch.Tensor,
                                   modality_names: List[str],
                                   title: str = 'Modality Winner Map per Channel',
                                   batch_idx: int = 0) -> np.ndarray:
        """
        다중 채널 '승자 인덱스' 맵의 시각화 이미지를 생성하여,
        파일로 저장하는 대신 NumPy 배열로 반환합니다.
        """
        if winner_indices.dim() == 4:
            winner_indices = winner_indices[batch_idx]
        
        indices_np = winner_indices.detach().cpu().numpy()
        C, H, W = indices_np.shape

        ncols = int(np.ceil(np.sqrt(C)))
        nrows = int(np.ceil(C / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
        axes = axes.flatten()
        
        legend_str = "Colors: "
        for i, name in enumerate(modality_names):
            if i < 3:
                legend_str += f"{['R', 'G', 'B'][i]}:{name}  "
        fig.suptitle(f'{title}\n({legend_str.strip()})', fontsize=16)

        color_palette = np.array([
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]
        ], dtype=np.uint8)
        
        for i in range(C):
            index_map = indices_np[i]
            rgb_image = color_palette[index_map]
            
            ax = axes[i]
            ax.imshow(rgb_image)
            ax.axis('off')
            ax.set_title(f"Ch: {i}", fontsize=10)
        
        for j in range(C, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Figure를 파일로 저장하는 대신 RGB 버퍼로 렌더링
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Figure 객체를 닫아 메모리 누수 방지
        plt.close(fig)
        
        return img_array
