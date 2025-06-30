import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from .clip import create_model

# Projector 模块，输出统一维度
class ClipProjector(nn.Module):
    def __init__(self, c_in, out_dim=768):
        super(ClipProjector, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(c_in, out_dim, bias=False),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)

# 仅使用 Projector 的消融版本
class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features

        # 为每一层添加 projector（对 token 特征）
        self.seg_projectors = nn.ModuleList([
            ClipProjector(c_in=1024, out_dim=768) for _ in features
        ])
        self.det_projectors = nn.ModuleList([
            ClipProjector(c_in=1024, out_dim=768) for _ in features
        ])

        # 对最终 pooled token 添加 projector
        self.final_projector = ClipProjector(c_in=768, out_dim=768) # Corrected c_in

    def forward(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0],
                    1, x.shape[-1], dtype=x.dtype, device=x.device),x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)

        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(24):
            x, _ = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)

            if (i + 1) in self.features:
                feature = x.permute(1, 0, 2)  # (B, L, C)

                seg_projected = self.seg_projectors[self.features.index(i + 1)](feature)
                det_projected = self.det_projectors[self.features.index(i + 1)](feature)

                seg_patch_tokens.append(seg_projected)
                det_patch_tokens.append(det_projected)

        x = x.permute(1, 0, 2)
        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        projected_cls = self.final_projector(pooled)

        return projected_cls, seg_patch_tokens, det_patch_tokens

# 主函数：使用 ProjectorOnly 的模型版本
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建基础模型
    model = create_model('ViT-L-14-336', 240).to(device)

    # 替换为 Projector-only 消融模型
    model1 = CLIP_Inplanted(model, [6, 12, 18, 24]).to(device)

    # 打印模型结构
    from torchsummary import summary
    print(summary(model1, (3, 336, 336)))
