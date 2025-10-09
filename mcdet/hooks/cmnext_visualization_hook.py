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
import torch.nn as nn
from torch import Tensor
from mmengine.dist import master_only
try:
    import wandb
except ImportError:
    wandb = None


def create_soft_mix_contribution_map(
    attention_map: Tensor,
    batch_idx: int = 0
) -> np.ndarray:
    """
    Soft-Mix의 결과인 attention_map을 입력받아, 각 모달리티의 기여도를
    RGB 채널에 매핑한 NumPy 이미지 배열을 생성합니다.
    Args:
        attention_map (Tensor): (B, num_modals, H, W) 크기의 attention weight 텐서.
        batch_idx (int): 시각화할 배치의 인덱스.
    Returns:
        np.ndarray: (H, W, 3) 형태의 uint8 RGB 이미지 배열.
    """
    import numpy as np
    import torch
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
    # (보조 모달리티가 3개 초과 시, 추가 모달리티는 시각화에서 제외됨)
    # 0~1 범위의 float 값을 0~255 범위의 uint8 값으로 변환
    image_array = (contribution_rgb.numpy() * 255).astype(np.uint8)
    return image_array

def log_soft_mix_stats(attention_map: Tensor, modality_names: List[str], log_prefix: str, batch_idx: int = 0):
    if wandb is None or attention_map is None:
        return
    # 텍스트 통계 로깅
    avg_contributions = attention_map[batch_idx].mean(dim=[1, 2])
    stats_dict = {}
    for i, name in enumerate(modality_names):
        stats_dict[f"{log_prefix}_contribution/{name}"] = avg_contributions[i].item()
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(stats_dict)
    except Exception:
        pass
    # 시각화 맵 이미지 생성 및 로깅
    image_array = create_soft_mix_contribution_map(attention_map, modality_names, "Contribution Map", batch_idx)
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({f"{log_prefix}_contribution_map": wandb.Image(image_array)})
    except Exception:
        pass
    
    
    

def vis_tensor_grid(tensor: torch.Tensor, batch=0, save_path='tmp_grid_colormap.png'):
    """
    Visualize all 12 channels of a tensor in a 4x3 grid with Grad-CAM style colormap (jet).
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) or (B, C, H, W).
        batch (int): Batch index if tensor has a batch dimension.
        save_path (str): Path to save the grid image.
    """
    import matplotlib.pyplot as plt
    # Handle batch dimension
    if tensor.dim() == 4:  # (B, C, H, W)
        tensor = tensor[batch]
    C, H, W = tensor.shape
    assert C > 12, "Tensor must have 12 channels"
    tensor_np = tensor.detach().cpu().numpy()
    # Normalize each channel to [0,1] for better colormap visualization
    tensor_np = (tensor_np - tensor_np.min(axis=(1,2), keepdims=True)) / \
                (tensor_np.max(axis=(1,2), keepdims=True) - tensor_np.min(axis=(1,2), keepdims=True) + 1e-8)
    # Create 4x3 subplot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i in range(12):
        r, c = divmod(i, 4)
        axes[r, c].imshow(tensor_np[i], cmap='jet')
        axes[r, c].axis('off')
        axes[r, c].set_title(f'Channel {i}')    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved 12-channel Grad-CAM style grid to {save_path}")
    
    
    
def vis_tensor_single_batch(tensor:torch.tensor, batch=0, save_path='vis_tensor.png'):
    import matplotlib.pyplot as plt
    import numpy as np
    tensor = tensor[batch]
    if len(tensor.shape) != 3:
        assert "Tensor must have 3 dimensions (C, H, W)"
    C, H, W = tensor.shape
    tensor_np = tensor.detach().cpu().numpy()
    tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min() + 1e-8)  # Normalize to [0, 1]
    tensor_np = (tensor_np * 255).astype(np.uint8)  # Convert to uint8 for visualization
    # Create a grid of images
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(tensor_np.transpose(1, 2, 0))  # Transpose to (H, W, C)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    
def vis_tensor_single_batch_grid(tensor:torch.tensor, batch=0, save_path='vis_tensor.png'):
    import matplotlib.pyplot as plt
    import numpy as np
    
    tensor = tensor[batch]
    tensor_np = tensor.detach().cpu().numpy()
    C, H, W = tensor_np.shape
    ncols = int(np.ceil(np.sqrt(C)))
    nrows = int(np.ceil(C / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten()
    
    for i in range(C):
        ch = tensor_np[i]
        axes[i].imshow(ch, cmap='jet')
        axes[i].axis('off')
        axes[i].set_title(f"Channel {i}", fontsize=10)
    
    # Hide unused axes
    for j in range(C, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved tensor visualization grid to {save_path}")

def log_soft_mix_stats(
        attention_map: Tensor,
        modality_names: List[str],
        log_prefix: str,
        batch_idx: int = 0
        ):
    """
    Soft Mix의 기여도 통계를 계산하고 시각화하여 wandb에 로깅합니다.
    """

    if wandb is None or attention_map is None:
        return

    # --- 1. 텍스트 통계 로깅 (평균 기여도) ---
    # attention_map shape: (B, num_modals, H, W)
    # --- 1. 텍스트 통계 로깅 ---
    avg_contributions = attention_map[batch_idx].mean(dim=[1, 2])
    stats_dict = {}
    for i, name in enumerate(modality_names):
        stats_dict[f"{log_prefix}_contribution/{name}"] = avg_contributions[i].item()
    
    # wandb가 초기화된 경우에만 로깅
    try:
        if wandb.run is not None:
            wandb.log(stats_dict)
    except Exception:
        pass

    # --- 2. 시각화 맵 로깅 (RGB 기여도 맵) ---
    # 각 모달리티의 기여도를 R, G, B 채널에 매핑합니다.
    # (주의: 보조 모달리티가 3개일 때 가장 직관적입니다)
    num_modals = attention_map.shape[1]
    if num_modals >= 3:
        # (num_modals, H, W) -> (H, W, num_modals) -> (H, W, 3)
        contribution_rgb = attention_map[batch_idx][:3].permute(1, 2, 0)
    elif num_modals == 2:
        # R, G 채널만 사용하고 B는 0으로 채웁니다.
        zeros = torch.zeros_like(attention_map[batch_idx][0])
        contribution_rgb = torch.stack([attention_map[batch_idx][0], attention_map[batch_idx][1], zeros], dim=-1)
    else: # num_modals == 1
        # R 채널만 사용 (흑백)
        ch = attention_map[batch_idx][0]
        contribution_rgb = torch.stack([ch, ch, ch], dim=-1)

    # 0~1 범위를 0~255 범위의 이미지로 변환
    color_map_np = (contribution_rgb * 255).byte().cpu().numpy()
    
    # wandb에 이미지 로깅
    caption = ""
    for i, name in enumerate(modality_names):
        if i < 3:
            caption += f"{['R', 'G', 'B'][i]}:{name} "
            
    try:
        if wandb.run is not None:
            wandb.log({f"{log_prefix}_contribution_map": wandb.Image(color_map_np, caption=caption.strip())})
    except Exception:
        pass
    



def _create_tensor_grid_image(tensor: torch.Tensor, title: str, batch_idx: int = 0) -> np.ndarray:
    # (이전 답변의 create_tensor_grid_image 코드 전체를 여기에 붙여넣기)
    import matplotlib.pyplot as plt
    
    if tensor.dim() == 4:
        tensor = tensor[batch_idx]
    
    tensor_np = tensor.detach().cpu().numpy()
    C, H, W = tensor_np.shape

    ncols = int(np.ceil(np.sqrt(C)))
    nrows = int(np.ceil(C / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    if C == 1:
        axes = [axes]
    else:
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
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_array

def _create_soft_mix_contribution_map(attention_map: torch.Tensor, batch_idx: int = 0) -> np.ndarray:
    # (이전 답변의 create_soft_mix_contribution_map 코드 전체를 여기에 붙여넣기)
    attention_map_single = attention_map[batch_idx].cpu()
    num_modals, h, w = attention_map_single.shape
    contribution_rgb = torch.zeros((h, w, 3), dtype=torch.float32)

    if num_modals >= 1: contribution_rgb[:, :, 0] = attention_map_single[0]
    if num_modals >= 2: contribution_rgb[:, :, 1] = attention_map_single[1]
    if num_modals >= 3: contribution_rgb[:, :, 2] = attention_map_single[2]
        
    image_array = (contribution_rgb.numpy() * 255).astype(np.uint8)
    return image_array

@HOOKS.register_module()
class CMNeXtVisualizationHook(Hook):
    def __init__(self, interval: int = 1):
        self.interval = interval

    @master_only
    def _after_val_epoch(self, runner: Runner) -> None:
        if (runner.epoch + 1) % self.interval != 0:
            return

        if wandb is None or wandb.run is None:
            runner.logger.warning('Wandb is not initialized. Skipping CMNeXtVisualizationHook.')
            return

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        
        backbone = model.backbone.cmnext_model.backbone
        if not hasattr(backbone, 'visualization_buffer') or not backbone.visualization_buffer:
            return
            
        buffer = backbone.visualization_buffer
        log_data = {}
        
        for stage, tensors in buffer.items():
            log_prefix = f"val_epoch_{runner.epoch+1}/{stage}"
            
            # ✅ [수정] 스코어 맵 시각화 로직 호출
            if tensors.get('scores') is not None:
                self._log_modality_scores(
                    tensors['scores'], backbone.modals, log_prefix, log_data)

            if tensors.get('attention_weights') is not None:
                self._log_soft_mix_stats(
                    tensors['attention_weights'], backbone.modals, log_prefix, log_data)

            feature_maps_to_log = {
                'RGB_Feature': tensors.get('rgb_feature'),
                'Fused_Aux_Feature': tensors.get('fused_aux_feature'),
                'Final_Fused_Feature': tensors.get('final_fused_feature')
            }
            self._log_feature_maps(feature_maps_to_log, log_prefix, log_data)
        
        if log_data:
            wandb.log(log_data)
            runner.logger.info(f"Logged CMNeXt feature maps for epoch {runner.epoch+1} to Wandb.")
        
        backbone.visualization_buffer.clear()
        
    def _log_modality_scores(self, scores, modality_names, log_prefix, log_data):
        """[✅ 새로 추가] 각 모달리티의 스코어 맵을 시각화하여 로깅합니다."""
        # scores는 (B, 1, H, W) 형태의 텐서 리스트
        for i, (score_map, name) in enumerate(zip(scores, modality_names)):
            image_array = _create_tensor_grid_image(
                score_map, title=f"{name} Score Map")
            log_data[f"{log_prefix}_score_map/{name}"] = wandb.Image(image_array, caption=f"{name} Score Map")

    def _log_soft_mix_stats(self, attention_map, modality_names, log_prefix, log_data):
        avg_contributions = attention_map[0].mean(dim=[1, 2])
        for i, name in enumerate(modality_names):
            log_data[f"{log_prefix}_contribution/{name}"] = avg_contributions[i].item()
        
        image_array = _create_soft_mix_contribution_map(attention_map)
        caption = " ".join([f"{c}:{n}" for c, n in zip(['R', 'G', 'B'], modality_names[:3])])
        log_data[f"{log_prefix}_contribution_map"] = wandb.Image(image_array, caption=caption)

    def _log_feature_maps(self, feature_maps, log_prefix, log_data):
        for name, tensor in feature_maps.items():
            if tensor is not None:
                image_array = _create_tensor_grid_image(tensor, title=f"{name}")
                log_data[f"{log_prefix}_{name}"] = wandb.Image(image_array)