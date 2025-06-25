# mcdet/models/backbones/stitchfusion.py

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
from typing import Dict, List, Union, Optional, Tuple

from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
import functools
from functools import partial
import warnings


def load_stitchfusion_pretrained(model, model_file):
    """Load pretrained SegFormer weights for StitchFusion."""
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
        if 'state_dict' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['state_dict']
    else:
        raw_state_dict = model_file
    
    state_dict = {}
    for k, v in raw_state_dict.items():
        # Remove common prefixes
        key = k.replace('backbone.', '').replace('encoder.', '')
        
        # Keep SegFormer/MiT related keys
        if any(keyword in key for keyword in ['patch_embed', 'block', 'norm', 'layer']):
            state_dict[key] = v

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[StitchFusion] Pretrained SegFormer model loaded: {msg}")
    del state_dict


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)

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


class MultiAdapter(nn.Module):
    """
    Multi-directional Adapter for cross-modal information transfer.
    This is the core component of StitchFusion that enables information 
    exchange between frozen encoders of different modalities.
    """
    
    def __init__(self, 
                 dim: int,
                 num_modals: int = 2,
                 reduction: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_modals = num_modals
        self.reduction = reduction
        
        # Multi-directional MLP layers for cross-modal information exchange
        self.cross_modal_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // reduction),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim // reduction, dim),
                nn.Dropout(dropout)
            ) for _ in range(num_modals)
        ])
        
        # Attention weights for adaptive fusion
        self.attention_weights = nn.Parameter(torch.ones(num_modals, num_modals))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_modals)
        ])

    def forward(self, modal_features: List[Tensor]) -> List[Tensor]:
        """
        Cross-modal information transfer between different modalities.
        
        Args:
            modal_features: List of feature tensors from different modalities
                          Each tensor shape: (B, H*W, C)
        Returns:
            List of enhanced feature tensors with cross-modal information
        """
        if len(modal_features) != self.num_modals:
            # Pad with zeros if fewer modalities provided
            while len(modal_features) < self.num_modals:
                modal_features.append(torch.zeros_like(modal_features[0]))
        
        enhanced_features = []
        attention_weights = F.softmax(self.attention_weights, dim=1)
        
        for i in range(self.num_modals):
            # Start with original feature
            enhanced_feat = modal_features[i]
            
            # Cross-modal information integration
            cross_modal_info = torch.zeros_like(enhanced_feat)
            
            for j in range(self.num_modals):
                if i != j:
                    # Transform other modality's features
                    transformed_feat = self.cross_modal_mlp[j](modal_features[j])
                    # Weighted aggregation
                    cross_modal_info += attention_weights[i, j] * transformed_feat
            
            # Residual connection + layer norm
            enhanced_feat = enhanced_feat + cross_modal_info
            enhanced_feat = self.layer_norms[i](enhanced_feat)
            
            enhanced_features.append(enhanced_feat)
        
        return enhanced_features


class Attention(nn.Module):
    """SegFormer-style efficient attention with spatial reduction."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DWConv(nn.Module):
    """Depth-wise convolution for Feed Forward Network."""
    
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """Feed Forward Network with depth-wise convolution."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """SegFormer transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping patches."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MixVisionTransformer(nn.Module):
    """
    Mix Vision Transformer (MiT) from SegFormer.
    This is the frozen encoder that will be used for each modality.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


# SegFormer model configurations
mit_settings = {
    'B0': {
        'embed_dims': [32, 64, 160, 256], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [4, 4, 4, 4],
        'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6), 'depths': [2, 2, 2, 2],
        'sr_ratios': [8, 4, 2, 1], 'drop_rate': 0.0, 'drop_path_rate': 0.1
    },
    'B1': {
        'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [4, 4, 4, 4],
        'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6), 'depths': [2, 2, 2, 2],
        'sr_ratios': [8, 4, 2, 1], 'drop_rate': 0.0, 'drop_path_rate': 0.1
    },
    'B2': {
        'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [4, 4, 4, 4],
        'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6), 'depths': [3, 4, 6, 3],
        'sr_ratios': [8, 4, 2, 1], 'drop_rate': 0.0, 'drop_path_rate': 0.1
    },
    'B3': {
        'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [4, 4, 4, 4],
        'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6), 'depths': [3, 4, 18, 3],
        'sr_ratios': [8, 4, 2, 1], 'drop_rate': 0.0, 'drop_path_rate': 0.1
    },
    'B4': {
        'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [4, 4, 4, 4],
        'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6), 'depths': [3, 8, 27, 3],
        'sr_ratios': [8, 4, 2, 1], 'drop_rate': 0.0, 'drop_path_rate': 0.1
    },
    'B5': {
        'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8], 'mlp_ratios': [4, 4, 4, 4],
        'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6), 'depths': [3, 6, 40, 3],
        'sr_ratios': [8, 4, 2, 1], 'drop_rate': 0.0, 'drop_path_rate': 0.1
    }
}


class StitchFusion(nn.Module):
    """
    StitchFusion: Original architecture with frozen SegFormer encoders + MultiAdapter.
    
    This follows the original StitchFusion design:
    - Uses frozen pre-trained SegFormer/MiT encoders for each modality
    - MultiAdapter for cross-modal information exchange
    - Minimal additional parameters for efficient learning
    """
    
    def __init__(self, 
                 variant: str = 'B2',
                 modals: List[str] = ['rgb', 'depth', 'event', 'lidar'],
                 freeze_encoders: bool = True):
        super().__init__()
        
        assert variant in mit_settings.keys(), f"Variant should be in {list(mit_settings.keys())}"
        
        self.variant = variant
        self.modals = modals
        self.num_modals = len(modals)
        self.freeze_encoders = freeze_encoders
        
        # Get model configuration
        config = mit_settings[variant]
        self.embed_dims = config['embed_dims']
        
        # Create independent SegFormer encoder for each modality
        self.encoders = nn.ModuleList([
            MixVisionTransformer(**config) for _ in range(self.num_modals)
        ])
        
        # Freeze encoders as per original StitchFusion design
        if freeze_encoders:
            for encoder in self.encoders:
                encoder.requires_grad_(False)
                encoder.eval()
        
        # MultiAdapter modules for each stage (cross-modal information exchange)
        self.multi_adapters = nn.ModuleList([
            MultiAdapter(
                dim=self.embed_dims[i], 
                num_modals=self.num_modals,
                reduction=16,
                dropout=0.1
            ) for i in range(4)  # 4 stages in SegFormer
        ])
        
        # Only MultiAdapters are trainable in original design
        for adapter in self.multi_adapters:
            adapter.requires_grad_(True)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass following original StitchFusion design.
        
        Args:
            inputs: List of tensors for different modalities [RGB, Depth, Event, LiDAR]
                   Each tensor shape: (B, C, H, W)
        
        Returns:
            List of multi-scale fused features for detection
        """
        assert len(inputs) == self.num_modals, f"Expected {self.num_modals} inputs, got {len(inputs)}"
        
        # Extract features from each modality using frozen encoders
        modal_features = []
        for i, (input_tensor, encoder) in enumerate(zip(inputs, self.encoders)):
            # Set encoder to eval mode if frozen
            if self.freeze_encoders:
                encoder.eval()
            
            # Extract multi-scale features
            features = encoder(input_tensor)  # List of 4 feature maps
            modal_features.append(features)
        
        # Cross-modal fusion using MultiAdapter at each stage
        fused_features = []
        
        for stage_idx in range(4):  # 4 stages in SegFormer
            # Collect features from all modalities at current stage
            stage_features = []
            for modal_idx in range(self.num_modals):
                feat = modal_features[modal_idx][stage_idx]
                B, C, H, W = feat.shape
                # Convert to sequence format for MultiAdapter
                feat_seq = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
                stage_features.append(feat_seq)
            
            # Apply MultiAdapter for cross-modal information exchange
            enhanced_features = self.multi_adapters[stage_idx](stage_features)
            
            # Use the first modality (RGB) as the main output after fusion
            # In the original design, all modalities contribute but RGB is typically the primary output
            main_enhanced = enhanced_features[0]  # RGB enhanced with cross-modal info
            B, HW, C = main_enhanced.shape
            H = W = int(HW ** 0.5)  # Assume square feature maps
            
            # Convert back to spatial format
            fused_feat = main_enhanced.transpose(1, 2).reshape(B, C, H, W)
            fused_features.append(fused_feat)
        
        return fused_features

    def load_pretrained_encoders(self, pretrained_path: str):
        """Load pretrained SegFormer weights into all encoders."""
        for i, encoder in enumerate(self.encoders):
            load_stitchfusion_pretrained(encoder, pretrained_path)
            print(f"[StitchFusion] Loaded pretrained weights for encoder {i} ({self.modals[i]})")

    def freeze_encoders_layers(self):
        """Freeze all encoder parameters."""
        for encoder in self.encoders:
            encoder.requires_grad_(False)
            encoder.eval()
        print("[StitchFusion] All encoders frozen")

    def unfreeze_encoders_layers(self):
        """Unfreeze all encoder parameters."""
        for encoder in self.encoders:
            encoder.requires_grad_(True)
            encoder.train()
        print("[StitchFusion] All encoders unfrozen")


class StitchFusionBaseModel(BaseModule):
    """Base model wrapper for StitchFusion following original design."""
    
    def __init__(self, 
                 backbone: str = 'stitchfusion-B2', 
                 modals: List[str] = ['rgb', 'depth', 'event', 'lidar'],
                 freeze_encoders: bool = True,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        
        # Parse backbone name
        if '-' in backbone:
            _, variant = backbone.split('-')
        else:
            variant = 'B2'  # Default
            
        self.backbone = StitchFusion(variant, modals, freeze_encoders)
        self.modals = modals
        self.variant = variant

    def init_pretrained(self, pretrained: str = None) -> None:
        """Load pretrained SegFormer weights."""
        if pretrained:
            self.backbone.load_pretrained_encoders(pretrained)


@MODELS.register_module()
class StitchFusionBackbone(BaseModule):
    """
    StitchFusion Backbone for Object Detection (Original Architecture).
    
    This implementation follows the original StitchFusion design:
    - Frozen pre-trained SegFormer encoders for each modality  
    - MultiAdapter modules for cross-modal information exchange
    - Minimal additional parameters for efficient learning
    
    Args:
        backbone (str): Backbone variant, e.g., 'stitchfusion-B2'
        modals (list): List of modalities to process
        freeze_encoders (bool): Whether to freeze the SegFormer encoders
        out_indices (tuple): Output indices for FPN
        frozen_stages (int): Stages to be frozen
        norm_eval (bool): Whether to set norm layers to eval mode
        pretrained (str): Path to pretrained SegFormer weights
    """
    
    def __init__(self,
                 backbone: str = 'stitchfusion-B2',
                 modals: List[str] = ['rgb', 'depth', 'event', 'lidar'],
                 freeze_encoders: bool = True,
                 out_indices: Tuple[int] = (0, 1, 2, 3),
                 frozen_stages: int = -1,
                 norm_eval: bool = False,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[dict] = None):
        
        super().__init__(init_cfg=init_cfg)
        
        self.backbone_name = backbone
        self.modals = modals
        self.freeze_encoders = freeze_encoders
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        
        # Create StitchFusion model following original design
        self.stitchfusion_model = StitchFusionBaseModel(
            backbone=backbone, 
            modals=modals,
            freeze_encoders=freeze_encoders,
            init_cfg=init_cfg
        )
        
        # Determine output channels based on variant
        variant = backbone.split('-')[-1] if '-' in backbone else 'B2'
        if variant in mit_settings:
            self.out_channels = mit_settings[variant]['embed_dims']
        else:
            self.out_channels = [64, 128, 320, 512]  # Default B2
        
        # Load pretrained weights if provided
        if pretrained:
            self.stitchfusion_model.init_pretrained(pretrained)
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze specified stages or components."""
        if self.frozen_stages == -1:
            # Follow original design: encoders can be frozen, adapters trainable
            if self.freeze_encoders:
                self.stitchfusion_model.backbone.freeze_encoders_layers()
                print("🧊 Encoders frozen (original StitchFusion design)")
            else:
                print("🔥 All parameters trainable")
            return

        if self.frozen_stages == 999:
            # Freeze everything
            print("🧊 Freezing ALL parameters")
            for param in self.stitchfusion_model.parameters():
                param.requires_grad = False
            return
        
        # Custom freezing logic if needed
        print(f"🧊 Custom freezing for stages {self.frozen_stages}")
    
    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Forward pass of StitchFusion backbone.
        
        Args:
            x: List of multimodal tensors [rgb_tensor, depth_tensor, event_tensor, lidar_tensor]
               Each tensor has shape (B, C, H, W)
        
        Returns:
            Tuple of feature tensors from different stages
        """
        features = self.stitchfusion_model.backbone(x)
        
        # Select output features based on out_indices
        outs = [features[i] for i in self.out_indices if i < len(features)]
        
        return tuple(outs)
    
    def train(self, mode: bool = True):
        """Set train/eval mode while respecting frozen encoders."""
        super().train(mode)
        
        # Keep frozen encoders in eval mode
        if self.freeze_encoders and mode:
            for encoder in self.stitchfusion_model.backbone.encoders:
                encoder.eval()
        
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    m.eval()
    
    def init_weights(self):
        """Initialize weights following original design."""
        if self.init_cfg is None:
            # Only initialize MultiAdapter parameters
            for adapter in self.stitchfusion_model.backbone.multi_adapters:
                for m in adapter.modules():
                    if isinstance(m, nn.Linear):
                        trunc_normal_(m.weight, std=.02)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        else:
            super().init_weights()