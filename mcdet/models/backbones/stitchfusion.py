# mcdet/models/backbones/cmnext_simple.py

import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from semseg.models.layers import DropPath
import torch.nn.init as init
import numpy as np
from math import factorial
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from mcdet.models.modules.ffm import FeatureFusionModule as FFM
from mcdet.models.modules.mspa import MSPABlock
from mcdet.models.modules.ffm import FeatureRectifyModule as FRM
import functools
from functools import partial
import warnings
import torch.nn.functional as F
import math
from typing import Dict, List, Union, Optional, Tuple


from mcdet.models.backbones.cmnext import MLP, DWConv, Attention, PatchEmbed

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor

class CustomDWConv(nn.Module):

    def __init__(self, dim, kernel):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)

        # Apply Kaiming initialization with fan-in to the dwconv layer
        init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)
    
class CustomPWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)

        # Initialize pwconv layer with Kaiming initialization
        init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.bn(self.pwconv(x))
        return x.flatten(2).transpose(1, 2)

class Bi_direct_adapter(nn.Module):
    def __init__(self, dim, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(dim, 8)  
        self.adapter_mid = nn.Linear(8, 8)
        self.adapter_up = nn.Linear(8, dim)  

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)   
        #x_down = self.act(x_down)
        x_down = F.gelu(self.adapter_mid(x_down))
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up #.permute(0, 2, 1).reshape(B, C, H, W)

class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # Initialize linear layers with Kaiming initialization
        for m in self.fc:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return (x * y.expand_as(x)).flatten(2).transpose(1, 2)

class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = CustomPWConv(c2)
        self.dwconv3 = CustomDWConv(c2, 3)
        self.dwconv5 = CustomDWConv(c2, 5)
        self.dwconv7 = CustomDWConv(c2, 7)
        self.pwconv2 = CustomPWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

        # Initialize fc1 layer with Kaiming initialization
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        x = self.fc1(x)
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        x3 = self.dwconv7(x, H, W)
        return self.fc2(F.gelu(self.pwconv2(x + x1 + x2 + x3, H, W)))
    
class FeatureCross(nn.Module):
    def __init__(self, channels, num_modals):
        super(FeatureCross, self).__init__()
        self.channels = channels
        self.num_modals = num_modals

        self.liner_fusion_layers = nn.ModuleList([
            nn.Linear(self.channels[0]*self.num_modals, self.channels[0]),
            nn.Linear(self.channels[1]*self.num_modals, self.channels[1]),
            nn.Linear(self.channels[2]*self.num_modals, self.channels[2]),
            nn.Linear(self.channels[3]*self.num_modals, self.channels[3]),
        ])

        self.mix_ffn = nn.ModuleList([
            MixFFN(self.channels[0], self.channels[0]),
            MixFFN(self.channels[1], self.channels[1]),
            MixFFN(self.channels[2], self.channels[2]),
            MixFFN(self.channels[3], self.channels[3]),
        ])

        self.channel_attns = nn.ModuleList([
            ChannelAttentionBlock(self.channels[0]),
            ChannelAttentionBlock(self.channels[1]),
            ChannelAttentionBlock(self.channels[2]),
            ChannelAttentionBlock(self.channels[3]),
        ])

    def forward(self, x, layer_idx):
        B, C, H, W = x[0].shape
        # conv fusion
        x = torch.cat(x, dim=1)
        x = x.flatten(2).transpose(1, 2)
        x_sum = self.liner_fusion_layers[layer_idx](x)
        # x_sum = self.mix_ffn[layer_idx](x_sum, H, W)
        x_sum = self.mix_ffn[layer_idx](x_sum, H, W)  + self.channel_attns[layer_idx](x_sum, H, W)
        return x_sum.reshape(B, H, W, -1).permute(0, 3, 1, 2)

class FeatureConc(nn.Module):
    def __init__(self, channels, num_modals):
        super(FeatureConc, self).__init__()
        self.channels = channels
        self.num_modals = num_modals

        self.liner_fusion_layers = nn.ModuleList([
            nn.Linear(self.channels[0]*self.num_modals, self.channels[0]),
            nn.Linear(self.channels[1]*self.num_modals, self.channels[1]),
            nn.Linear(self.channels[2]*self.num_modals, self.channels[2]),
            nn.Linear(self.channels[3]*self.num_modals, self.channels[3]),
        ])

    def forward(self, x, layer_idx):
        B, C, H, W = x[0].shape
        # conv fusion
        x = torch.cat(x, dim=1)
        x = x.flatten(2).transpose(1, 2)
        x_sum = self.liner_fusion_layers[layer_idx](x)
        return x_sum.reshape(B, H, W, -1).permute(0, 3, 1, 2)

class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., drop_path=0., mlp_hidden_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = MLP(dim, mlp_hidden_dim)
        self.norm_v = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        q = q.softmax(-2).transpose(-1,-2)
        _, _, Nk, Ck  = k.shape
        k = k.softmax(-2)
        k = torch.nn.functional.avg_pool2d(k, (1, Ck))
        
        attn = self.sigmoid(q @ k)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd*Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x 

mit_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}

class Block_every_one(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., num_modalities=2, fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4)) if not fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim * 4))
        # Initialize adapters dynamically based on number of modalities
        for i in range(num_modalities):
            for j in range(num_modalities):  
                if i < j:
                    adap_t_att = Bi_direct_adapter(dim)  
                    adap_t_mlp = Bi_direct_adapter(dim)              
                    # ---------- saving in setattr
                    setattr(self, f"adap_t_att{i + 1}{j + 1}", adap_t_att)
                    setattr(self, f"adap_t_mlp{i + 1}{j + 1}", adap_t_mlp)

    def forward(self, inputs, H, W):
        
        outputs = [x.clone() for x in inputs]
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeAtt = outputs
        # 首先，每个输入独立经过attention处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # 其次，使用adap_t适配器进行模态间影响
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeAtt[i]  # 提取出原始的没有经过multi-attention的
            # 使用adap_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]
                    
                    if i < j:
                        adap_t_att = getattr(self, f"adap_t_att{i + 1}{j + 1}")
                    else:
                        adap_t_att = getattr(self, f"adap_t_att{j + 1}{i + 1}")

                    outputs[j] = xi + self.drop_path(adap_t_att(self.norm1(x_ori)))
        
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeMLP = outputs
        # 每个输入独立经过MLP处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # 再次，经过adap2_t适配器处理
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeMLP[i]
            # 使用adap2_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]

                    if i < j:
                        adap_t_mlp = getattr(self, f"adap_t_mlp{i + 1}{j + 1}")
                    else:
                        adap_t_mlp = getattr(self, f"adap_t_mlp{j + 1}{i + 1}")

                    outputs[j] = xi + self.drop_path(adap_t_mlp(self.norm2(x_ori)))

        return outputs

class Block_every_two(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., num_modalities=2, fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4)) if not fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim * 4))
 
        # Initialize adapters dynamically based on number of modalities
        for i in range(num_modalities):
            for j in range(num_modalities):  
                if i != j:
                    adap_t_att = Bi_direct_adapter(dim)  
                    adap_t_mlp = Bi_direct_adapter(dim)              
                    # ---------- saving in setattr
                    setattr(self, f"adap_t_att{i + 1}{j + 1}", adap_t_att)
                    setattr(self, f"adap_t_mlp{i + 1}{j + 1}", adap_t_mlp)

    def forward(self, inputs, H, W):
        outputs = [x.clone() for x in inputs]
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeAtt = outputs
        # 首先，每个输入独立经过attention处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # 其次，使用adap_t适配器进行模态间影响
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeAtt[i]  # 提取出原始的没有经过multi-attention的
            # 使用adap_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]
                    adap_t_att = getattr(self, f"adap_t_att{i + 1}{j + 1}")
                    outputs[j] = xi + self.drop_path(adap_t_att(self.norm1(x_ori)))
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeMLP = outputs
        # 每个输入独立经过MLP处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # 再次，经过adap2_t适配器处理
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeMLP[i]
            # 使用adap2_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]
                    adap_t_mlp = getattr(self, f"adap_t_mlp{i + 1}{j + 1}")
                    outputs[j] = xi + self.drop_path(adap_t_mlp(self.norm2(x_ori)))
        return outputs


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0. , num_modalities=2, fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4)) if not fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim * 4))

    def forward(self, inputs: list, H, W):
        outputs = [x.clone() for x in inputs]
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # 首先，每个输入独立经过attention处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # 每个输入独立经过MLP处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return outputs

# for MCUBES AD AN 请您在复现论文的时候考虑 StitchFusion实际上在两个模态的时候Block_shared和Block_every_one是相同的。
# 一些 .pth文件 是只能适配 Block_shared的。请您到时候辨别性的复现。
class Block_shared(nn.Module): 
    def __init__(self, dim, head, sr_ratio=1, dpr=0. , num_modalities=2, fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4)) if not fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim * 4))

        self.adap_t = Bi_direct_adapter(dim)
        self.adap2_t = Bi_direct_adapter(dim)

    def forward(self, inputs: list, H, W):
        outputs = [x.clone() for x in inputs]
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeAtt = outputs
        # 首先，每个输入独立经过attention处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # 其次，使用adap_t适配器进行模态间影响
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeAtt[i]  # 提取出原始的没有经过multi-attention的
            # 使用adap_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]
                    outputs[j] = xi + self.drop_path(self.adap_t(self.norm1(x_ori)))
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeMLP = outputs
        # 每个输入独立经过MLP处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # 再次，经过adap2_t适配器处理
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeMLP[i]
            # 使用adap2_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]
                    outputs[j] = xi + self.drop_path(self.adap2_t(self.norm2(x_ori)))
        return outputs

# ----------------------------------------- 总体框架 ----------------------------------------- #
class stitchfusion(nn.Module):
    def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):
        super().__init__()
        assert model_name in mit_settings.keys(), f"Model name should be in {list(mit_settings.keys())}"
        embed_dims, depths = mit_settings[model_name]
        self.modals = modals[1:] if len(modals)>1 else []  
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims
        self.num_stages = 4

        # ----------------------------------------------------------------- 定义RGB  
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        pano_1 = [1,2,5,8]
        pano_2 = [8,4,2,1]

        for i in range(self.num_stages):
            if i == 0 :
                cur = 0
            else:
                cur += depths[i-1]
            patch_embed = PatchEmbed(3 if i==0 else embed_dims[i-1], embed_dims[i], 7 if i == 0 else 3, 4 if i == 0 else 2, 7//2 if i == 0 else 3//2)
                
            
            # block = nn.ModuleList([Block(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j]) for j in range(depths[i])])

            # 1
            block = nn.ModuleList([Block_every_one(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j], self.num_modals+1) for j in range(depths[i])])
            # # 2
            # if i < 2 :
            #     block = nn.ModuleList([Block(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j]) for j in range(depths[i])])
            # else:
            #     block = nn.ModuleList([Block_every_one(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j], self.num_modals+1) for j in range(depths[i])])
            
            # 3
            # block = nn.ModuleList([Block_every_two(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j], self.num_modals+1) for j in range(depths[i])])

            # # 4
            # if i < 2 :
            #     block = nn.ModuleList([Block(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j]) for j in range(depths[i])])
            # else:
            #     block = nn.ModuleList([Block_every_two(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j], self.num_modals+1) for j in range(depths[i])])
            
            norm = nn.LayerNorm(embed_dims[i])
            
            # ---------- saving in setattr
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

        feature_cross = FeatureCross(self.channels, num_modals = self.num_modals + 1)
        setattr(self, f"feature_cross", feature_cross)
        
        # feature_conc = FeatureConc(self.channels, num_modals = self.num_modals + 1)
        # setattr(self, f"feature_conc", feature_conc)

    def forward(self, x: list) -> list:
        
        x_in = [t.clone() for t in x]
        B = x[0].shape[0]
        outs = []

        for i in range(self.num_stages):         
            # -------------- Transformer -------------- #
            # patch_embed
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            for kk in range(len(x)):
                x_in[kk], H, W = patch_embed(x_in[kk])
            # block
            block = getattr(self, f"block{i + 1}")
            for blk_num in range(len(block)): 
                x_in = block[blk_num](x_in, H, W)
            # norm
            norm = getattr(self, f"norm{i + 1}")
            for kk in range(len(x)):
                x_in[kk] = norm(x_in[kk]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
                
            #--------------- 收集融合特征 ---------------#
            # 1. 使用FFMs
            feature_cross = getattr(self, f"feature_cross")
            x_fusion = feature_cross(x_in, layer_idx=i)
            
            # 2. add 
            # x_fusion = torch.sum(torch.stack(x_in), dim=0)

            # -------------- saving 
            outs.append(x_fusion)

        return outs
    
@MODELS.register_module()
class StitchFusionBackbone(BaseModule):
    """StitchFusion backbone for multimodal object detection.
    
    This backbone processes multimodal inputs using StitchFusion approach
    with cross-modal adapters and feature fusion.
    
    Args:
        backbone (str): Backbone variant, e.g., 'StitchFusion-B0', 'StitchFusion-B2'
        modals (list): List of modalities to process
        out_indices (tuple): Output indices for FPN
        frozen_stages (int): Stages to be frozen
        norm_eval (bool): Whether to set norm layers to eval mode
        pretrained (str): Path to pretrained weights
        adapter_type (str): Type of adapter ('every_one', 'every_two', 'shared', 'none')
    """
    
    def __init__(self,
                 backbone: str = 'StitchFusion-B2',
                 modals: List[str] = ['rgb', 'depth', 'event', 'lidar'],
                 out_indices: Tuple[int] = (0, 1, 2, 3),
                 frozen_stages: int = -1,
                 norm_eval: bool = False,
                 pretrained: Optional[str] = None,
                 adapter_type: str = 'every_one',  # 'every_one', 'every_two', 'shared', 'none'
                 init_cfg: Optional[dict] = None):
        
        super().__init__(init_cfg=init_cfg)
        
        self.backbone_name = backbone
        self.modals = modals
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.adapter_type = adapter_type
        
        # Extract model variant from backbone name
        if 'B0' in backbone:
            model_variant = 'B0'
        elif 'B1' in backbone:
            model_variant = 'B1'
        elif 'B2' in backbone:
            model_variant = 'B2'
        elif 'B3' in backbone:
            model_variant = 'B3'
        elif 'B4' in backbone:
            model_variant = 'B4'
        elif 'B5' in backbone:
            model_variant = 'B5'
        else:
            model_variant = 'B2'  # Default
        
        # Create StitchFusion model
        self.stitchfusion_model = stitchfusion(
            model_name=model_variant, 
            modals=modals
        )
        
        # Set adapter type for blocks
        self._configure_adapter_type()
        
        # Determine output channels based on backbone variant
        embed_dims = mit_settings[model_variant][0]
        self.out_channels = embed_dims  # [64, 128, 320, 512] for B2
        
        # Load pretrained weights if provided
        if pretrained:
            self.init_pretrained(pretrained)
        
        self._freeze_stages()
    
    def _configure_adapter_type(self):
        """Configure the adapter type for all blocks."""
        for i in range(self.stitchfusion_model.num_stages):
            block = getattr(self.stitchfusion_model, f"block{i + 1}")
            
            # Get model settings
            model_variant = self.backbone_name.split('-')[-1] if '-' in self.backbone_name else 'B2'
            embed_dims, depths = mit_settings.get(model_variant, mit_settings['B2'])
            
            # Create new blocks based on adapter type
            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
            pano_1 = [1, 2, 5, 8]
            pano_2 = [8, 4, 2, 1]
            
            if i == 0:
                cur = 0
            else:
                cur += depths[i-1]
            
            if self.adapter_type == 'every_one':
                new_block = nn.ModuleList([
                    Block_every_one(embed_dims[i], pano_1[i], pano_2[i], 
                                  dpr[cur+j], self.stitchfusion_model.num_modals+1) 
                    for j in range(depths[i])
                ])
            elif self.adapter_type == 'every_two':
                new_block = nn.ModuleList([
                    Block_every_two(embed_dims[i], pano_1[i], pano_2[i], 
                                   dpr[cur+j], self.stitchfusion_model.num_modals+1) 
                    for j in range(depths[i])
                ])
            elif self.adapter_type == 'shared':
                new_block = nn.ModuleList([
                    Block_shared(embed_dims[i], pano_1[i], pano_2[i], 
                               dpr[cur+j], self.stitchfusion_model.num_modals+1) 
                    for j in range(depths[i])
                ])
            else:  # 'none'
                new_block = nn.ModuleList([
                    Block(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j]) 
                    for j in range(depths[i])
                ])
            
            setattr(self.stitchfusion_model, f"block{i + 1}", new_block)
    
    def _freeze_stages(self):
        """Freeze stages according to frozen_stages."""
        if self.frozen_stages >= 0:
            # Freeze patch embedding layers
            for i in range(min(self.frozen_stages + 1, self.stitchfusion_model.num_stages)):
                patch_embed = getattr(self.stitchfusion_model, f"patch_embed{i + 1}")
                for param in patch_embed.parameters():
                    param.requires_grad = False
                
                # Freeze blocks
                block = getattr(self.stitchfusion_model, f"block{i + 1}")
                for param in block.parameters():
                    param.requires_grad = False
                
                # Freeze norms
                norm = getattr(self.stitchfusion_model, f"norm{i + 1}")
                for param in norm.parameters():
                    param.requires_grad = False
    
    def init_pretrained(self, pretrained: str = None):
        """Load pretrained weights."""
        if pretrained:
            try:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                
                # Filter out incompatible keys
                model_dict = self.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint.items() 
                                 if k in model_dict and model_dict[k].shape == v.shape}
                
                model_dict.update(pretrained_dict)
                msg = self.load_state_dict(model_dict, strict=False)
                print(f"[StitchFusion] Pretrained model loaded: {msg}")
                
            except Exception as e:
                print(f"[StitchFusion] Failed to load pretrained weights: {e}")
    
    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Forward pass of StitchFusion backbone.
        
        Args:
            x: List of multimodal tensors [rgb_tensor, depth_tensor, event_tensor, lidar_tensor]
               Each tensor has shape (B, C, H, W)
        
        Returns:
            Tuple of feature tensors from different stages
        """
        # Ensure input is a list
        if not isinstance(x, list):
            x = [x]
        
        # StitchFusion backbone expects list of multimodal inputs
        features = self.stitchfusion_model(x)
        
        # Return features according to out_indices
        outs = [features[i] for i in self.out_indices if i < len(features)]
        
        return tuple(outs)
    
    def train(self, mode: bool = True):
        """Set train/eval mode."""
        super().train(mode)
        
        if mode and self.norm_eval:
            # Set norm layers to eval mode
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    m.eval()
    
    def init_weights(self):
        """Initialize weights of the backbone."""
        if self.init_cfg is None:
            # Apply custom weight initialization
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
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
            
            self.apply(_init_weights)
        else:
            super().init_weights()

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization (from timm library)"""
    try:
        from timm.models.layers import trunc_normal_
        return trunc_normal_(tensor, mean, std, a, b)
    except ImportError:
        # Fallback to normal initialization
        with torch.no_grad():
            tensor.normal_(mean, std)
            tensor.clamp_(min=a, max=b)
        return tensor