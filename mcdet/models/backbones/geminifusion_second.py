
# geminifusion.py

import torch
import torch.nn as nn
import math
import warnings
from torch import Tensor
from torch.nn import functional as F
from mcdet.models.backbones.geminifusionbackbone import GeminiFusionBackbone
from typing import Tuple, List
from mmdet.registry import MODELS
from mmengine.model import BaseModule

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

@MODELS.register_module()
class GeminiFusion_second(BaseModule):
    """GeminiFusion backbone for multi-modal detection.
    
    This backbone extracts features from multiple modalities and outputs
    multi-scale features for detection heads.
    """
    
    def __init__(
        self,
        backbone: str = "GeminiFusion-B2",
        modals: list = ["rgb", "depth", "event", "lidar"],
        drop_path_rate: float = 0.0,
        out_indices: Tuple[int] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        init_cfg: dict = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        
        # Parse backbone variant
        if '-' in backbone:
            backbone_name, variant = backbone.split("-")
        else:
            backbone_name = "GeminiFusion"
            variant = backbone
            
        # Initialize GeminiFusion backbone
        self.backbone = GeminiFusionBackbone(
            backbone=variant,
            modals=modals,
            drop_path_rate=drop_path_rate,
            num_modal=len(modals),
        )
        
        self.modals = modals
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.num_parallel = 2
        
        # Get feature dimensions from backbone
        # These should match your GeminiFusionBackbone output dimensions
        if variant == 'B0':
            self.embed_dims = [32, 64, 160, 256]
        elif variant == 'B1':
            self.embed_dims = [64, 128, 320, 512]
        elif variant == 'B2':
            self.embed_dims = [64, 128, 320, 512]
        elif variant == 'B3':
            self.embed_dims = [64, 128, 320, 512]
        elif variant == 'B4':
            self.embed_dims = [64, 128, 320, 512]
        elif variant == 'B5':
            self.embed_dims = [64, 128, 320, 512]
        else:
            raise ValueError(f"Unsupported variant: {variant}")
            
        # Apply weight initialization
        self.apply(self._init_weights)
        
        # Freeze stages if specified
        self._freeze_stages()

    def forward(self, x: List[Tensor]) -> Tuple[Tensor]:
        """Forward function for backbone.
        
        Args:
            x: List of input tensors for different modalities
            
        Returns:
            Tuple of output feature maps at different scales
        """
        # Extract multi-modal features using GeminiFusion backbone
        x_modals = self.backbone(x)
        
        # x_modals should be a list of [branch1_features, branch2_features]
        # Each branch contains multi-scale features
        
        # For detection, we typically use the first branch or ensemble them
        # Here we'll use the first branch and return the specified indices
        if isinstance(x_modals, (list, tuple)) and len(x_modals) >= 1:
            # Use second branch features
            features = x_modals[1]
        else:
            features = x_modals
            
        # Select output features based on out_indices
        outs = []
        for i in self.out_indices:
            if i < len(features):
                outs.append(features[i])
            else:
                raise IndexError(f"Index {i} out of range for features of length {len(features)}")
        
        return tuple(outs)

    def _freeze_stages(self):
        """Freeze specified stages."""
        if self.frozen_stages >= 0:
            # Freeze backbone parameters up to specified stage
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        """Initialize backbone with pretrained weights."""
        if pretrained:
            checkpoint = torch.load(pretrained, map_location="cpu")
            if "state_dict" in checkpoint.keys():
                checkpoint = checkpoint["state_dict"]
            if "model" in checkpoint.keys():
                checkpoint = checkpoint["model"]
                
            # Remove head weights if they exist
            keys_to_remove = [k for k in checkpoint.keys() if 'head' in k]
            for key in keys_to_remove:
                checkpoint.pop(key, None)
                
            # Expand state dict for parallel structure
            checkpoint = self._expand_state_dict(
                self.backbone.state_dict(), checkpoint, self.num_parallel
            )
            msg = self.backbone.load_state_dict(checkpoint, strict=False)
            print(f"Pretrained weights loaded: {msg}")

    def _expand_state_dict(self, model_dict, state_dict, num_parallel):
        """Expand state dictionary for parallel structure."""
        model_dict_keys = model_dict.keys()
        state_dict_keys = state_dict.keys()
        
        for model_dict_key in model_dict_keys:
            model_dict_key_re = model_dict_key.replace("module.", "")
            if model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
            
            for i in range(num_parallel):
                ln = f".ln_{i}"
                replace = True if ln in model_dict_key_re else False
                model_dict_key_re = model_dict_key_re.replace(ln, "")
                if replace and model_dict_key_re in state_dict_keys:
                    model_dict[model_dict_key] = state_dict[model_dict_key_re]
        
        return model_dict

    def train(self, mode=True):
        """Override train mode to handle frozen stages."""
        super().train(mode)
        self._freeze_stages()
        return self


# Additional utility classes that might be needed
class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))