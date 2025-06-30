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

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels)
        x_trans = x.transpose(1, 2)  # (batch_size, channels, seq_len)
        avg_out = self.fc(self.avg_pool(x_trans).squeeze(-1))
        max_out = self.fc(self.max_pool(x_trans).squeeze(-1))
        out = self.sigmoid(avg_out + max_out)
        return out.unsqueeze(1)  # (batch_size, 1, channels)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels)
        x_trans = x.transpose(1, 2)  # (batch_size, channels, seq_len)
        avg_out = torch.mean(x_trans, dim=1, keepdim=True)
        max_out, _ = torch.max(x_trans, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out).transpose(1, 2)  # (batch_size, seq_len, 1)


# 添加频域注意力模块
class FrequencyAttention(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels)
        x_trans = x.transpose(1, 2)  # (batch_size, channels, seq_len)

        # 应用FFT
        fft_features = torch.fft.fft2(x_trans.float())
        magnitude = torch.abs(fft_features)

        # 提取频域特征
        freq_avg = torch.mean(magnitude, dim=-1)  # 平均频率响应
        freq_max, _ = torch.max(magnitude, dim=-1)  # 最大频率响应

        # 合并频域特征
        freq_info = (freq_avg + freq_max) / 2

        # 通过FC层
        freq_attention = self.fc(freq_info)
        freq_attention = self.sigmoid(freq_attention)

        return freq_attention.unsqueeze(1)  # (batch_size, 1, channels)


# 修改ClipAdapter类
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

        # 添加三种注意力模块
        self.channel_attention = ChannelAttention(c_in)
        self.spatial_attention = SpatialAttention()
        self.frequency_attention = FrequencyAttention(c_in)

    def forward(self, x):
        # 空间域注意力
        ca_out = self.channel_attention(x) * x  # 通道注意力
        sa_out = self.spatial_attention(ca_out) * ca_out  # 空间注意力

        # 频域注意力
        fa_out = self.frequency_attention(x) * x  # 频域注意力

        # 将所有注意力特征通过fc1降维到bottleneck维度
        ca_out_reduced = self.fc1(ca_out)
        sa_out_reduced = self.fc1(sa_out)
        fa_out_reduced = self.fc1(fa_out)

        # 原始Adapter路径
        adapter_x_original = self.fc1(x)

        # 在bottleneck维度上进行特征融合
        adapter_x = 0.4 * adapter_x_original + \
                    0.2 * ca_out_reduced + \
                    0.2 * sa_out_reduced + \
                    0.2 * fa_out_reduced

        # 经过第二个全连接层升维回原始维度
        adapter_y = self.fc2(adapter_x)

        # 最终输出使用较强的注意力融合
        y = 0.3 * adapter_y + \
            0.2 * ca_out + \
            0.2 * sa_out + \
            0.3 * fa_out

        return adapter_x, y


# 基于 CLIP 模型的扩展
# 给视觉编码器增加adapter
# 只做到了图片的特征提取————>返回四个中间级的特征包括分割和分类
class CLIP_Inplanted(nn.Module):
    # features是在哪个transformer层上增加适配器，这是一个列表层
    def __init__(self, clip_model, features):
        super().__init__()
        print("你使用的是不含 classname 的 CLIP_Inplanted")
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        # bottleneck 中间的特征，最后c_in=c_out
        # seg_adapters是一个列表具有多个ClipAdapter的列表
        # det_adapters是一个modulelist的列表具有多个ClipAdapter的列表
        # 在这里的长度是4个
        self.seg_adapters = nn.ModuleList([ClipAdapter(
            1024, bottleneck=768) for i in range(len(features))])

        self.det_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])

    def forward(self, x):

        # 将输入通过图像编码器的第一个卷积层
        x = self.image_encoder.conv1(x)
        # 将卷积层的输出重塑为 [batch_size, channels, num_patches]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # 在重塑为将卷积层的输出重塑为 [batch_size, num_patches, channels]
        x = x.permute(0, 2, 1)
        # 在第1维度上拼接别嵌入（class embedding）和卷积层的输出
        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0],
                                                                          1, x.shape[-1], dtype=x.dtype,
                                                                          device=x.device), x], dim=1)
        # 将位置嵌入（positional,embedding）加到输入上。
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        # 丢弃一些patch
        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        # 将张量的维度重新排列为 [num_patches, batch_size, channels]
        x = x.permute(1, 0, 2)

        attn_out = []
        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(24):

            # 保留一层的输出注意力，来获得其维度
            ############################################################################################
            if i + 1 == 12:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
                attn_out.append(attn)
            #############################################################################################

            # 这里是先经过vit
            else:
                x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)

            # 四个patch特征,放到6，12   18，24层的vit  添加适配器
            if (i + 1) in self.features:
                # 这里的seg_adapt_med,det_adapt_med的维度是(L,B,C)
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i + 1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i + 1)](x)

                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out
                # 这里的seg_patch_tokens,det_patch_tokens内元素的维度是(L,B,C)
                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

        # 这里感觉没有用
        #########################################################################################
        # # 获得特征的图的维度：（B，C，L）
        # # attn_out 只保留了12层的注意力图
        # B, C, L = attn_out[0].shape
        # # 计算
        # H = int(math.sqrt(L-1))
        # out_attn = torch.zeros([H, H]).to('cuda')
        #
        # for i in range(len(attn)):
        #     out_attn = out_attn + attn_out[i][0, 0, 1:].view(H, H)

        ##########################################################################################

        # 将张量的维度重新排列为[batch_size, num_patches, channels]
        x = x.permute(1, 0, 2)

        # 将中间级的维度重新排列为[batch_size, num_patches, channels] BLC
        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        # 返回四个中间级特征
        return pooled, seg_patch_tokens, det_patch_tokens


if __name__ == '__main__':
    # 设置设备，优先使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建基础的ViT模型，这里假设你有一个 create_model 函数用于创建模型
    model = create_model('ViT-L-14-336', 240).to(device)

    # 将基础模型传入 CLIP_Inplanted 以进行扩展
    model1 = CLIP_Inplanted(model, [6, 12, 18, 24]).to(device)

    # 打印模型结构摘要
    from torchsummary import summary

    print(summary(model1, (3, 336, 336)))

# import os
# import argparse
# import random
# import math
# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
# from PIL import Image
# from .clip import create_model
# from .promptlearner import PromptLearner
#
# # 添加CBAM模块
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x shape: (batch_size, seq_len, channels)
#         x_trans = x.transpose(1, 2)  # (batch_size, channels, seq_len)
#         avg_out = self.fc(self.avg_pool(x_trans).squeeze(-1))
#         max_out = self.fc(self.max_pool(x_trans).squeeze(-1))
#         out = self.sigmoid(avg_out + max_out)
#         return out.unsqueeze(1)  # (batch_size, 1, channels)
#
# # class ChannelAttention(nn.Module):
# #     def __init__(self, in_channels, reduction_ratio=16):
# #         super(ChannelAttention, self).__init__()
# #         self.avg_pool = nn.AdaptiveAvgPool1d(1)
# #         self.max_pool = nn.AdaptiveMaxPool1d(1)
# #
# #         self.fc = nn.Sequential(
# #             nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
# #             nn.ReLU(inplace=True),
# #             nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
# #         )
# #         self.sigmoid = nn.Sigmoid()
# #
# #     def forward(self, x):
# #         # x shape: (batch_size, seq_len, channels)
# #         x_trans = x.transpose(1, 2)  # (batch_size, channels, seq_len)
# #         avg_out = self.fc(self.avg_pool(x_trans).squeeze(-1))
# #         max_out = self.fc(self.max_pool(x_trans).squeeze(-1))
# #         out = self.sigmoid(avg_out + max_out)
# #
# #         # 应用Hadamard积
# #         return x * out.unsqueeze(1)  # (batch_size, seq_len, channels)
#
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x shape: (batch_size, seq_len, channels)
#         x_trans = x.transpose(1, 2)  # (batch_size, channels, seq_len)
#         avg_out = torch.mean(x_trans, dim=1, keepdim=True)
#         max_out, _ = torch.max(x_trans, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.conv(out)
#         return self.sigmoid(out).transpose(1, 2)  # (batch_size, seq_len, 1)
#
# # 添加频域注意力模块
# class FrequencyAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(FrequencyAttention, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // 16, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // 16, in_channels, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x shape: (batch_size, seq_len, channels)
#         x_trans = x.transpose(1, 2)  # (batch_size, channels, seq_len)
#
#         # 应用FFT
#         fft_features = torch.fft.fft2(x_trans.float())
#         magnitude = torch.abs(fft_features)
#
#         # 提取频域特征
#         freq_avg = torch.mean(magnitude, dim=-1)  # 平均频率响应
#         freq_max, _ = torch.max(magnitude, dim=-1)  # 最大频率响应
#
#         # 合并频域特征
#         freq_info = (freq_avg + freq_max) / 2
#
#         # 通过FC层
#         freq_attention = self.fc(freq_info)
#         freq_attention = self.sigmoid(freq_attention)
#
#         return freq_attention.unsqueeze(1)  # (batch_size, 1, channels)
#
# # 修改ClipAdapter类
# class ClipAdapter(nn.Module):
#     def __init__(self, c_in, bottleneck=768):
#         super(ClipAdapter, self).__init__()
#         self.fc1 = nn.Sequential(
#             nn.Linear(c_in, bottleneck, bias=False),
#             nn.LeakyReLU(inplace=False)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(bottleneck, c_in, bias=False),
#             nn.LeakyReLU(inplace=False)
#         )
#
#         # 添加三种注意力模块
#         self.channel_attention = ChannelAttention(c_in)
#         self.spatial_attention = SpatialAttention()
#         self.frequency_attention = FrequencyAttention(c_in)
#
#     def forward(self, x):
#         # 空间域注意力
#         ca_out = self.channel_attention(x) * x  # 通道注意力
#         sa_out = self.spatial_attention(ca_out) * ca_out  # 空间注意力
#
#         # 频域注意力
#         fa_out = self.frequency_attention(x) * x  # 频域注意力
#
#         # 将所有注意力特征通过fc1降维到bottleneck维度
#         ca_out_reduced = self.fc1(ca_out)
#         sa_out_reduced = self.fc1(sa_out)
#         fa_out_reduced = self.fc1(fa_out)
#
#         # 原始Adapter路径
#         adapter_x_original = self.fc1(x)
#
#         # 在bottleneck维度上进行特征融合
#         adapter_x = 0.4 * adapter_x_original + \
#                    0.2 * ca_out_reduced + \
#                    0.2 * sa_out_reduced + \
#                    0.2 * fa_out_reduced
#
#         # 经过第二个全连接层升维回原始维度
#         adapter_y = self.fc2(adapter_x)
#
#         # 最终输出使用较强的注意力融合
#         y = 0.3 * adapter_y + \
#             0.2 * ca_out + \
#             0.2 * sa_out + \
#             0.3 * fa_out
#
#         return adapter_x, y
# #########################################################################################################################
# class CrossModalAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=8):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
#
#     def forward(self, text_features, image_tokens):
#         attn_output, _ = self.cross_attn(text_features, image_tokens, image_tokens)
#         return attn_output

# 基于 CLIP 模型的扩展
# 给视觉编码器增加adapter
# 只做到了图片的特征提取————>返回四个中间级的特征包括分割和分类
# class CLIP_Inplanted(nn.Module):
#     # features是在哪个transformer层上增加适配器，这是一个列表层
#     def __init__(self, clip_model, features,text_features):
#     #def __init__(self, clip_model, features):
#         super().__init__()
#         self.clipmodel = clip_model
#         self.image_encoder = clip_model.visual
#         self.features = features
#
#         self.cross_attn = CrossModalAttention(embed_dim=768)
#
#         self.text_features = text_features
#
#         # bottleneck 中间的特征，最后c_in=c_out
#         # seg_adapters是一个列表具有多个ClipAdapter的列表
#         # det_adapters是一个modulelist的列表具有多个ClipAdapter的列表
#         # 在这里的长度是4个
#         self.seg_adapters = nn.ModuleList( [ClipAdapter(
#             1024, bottleneck=768) for i in range(len(features))] )
#
#         self.det_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )
#
#         #self.promptlearner = PromptLearner(input_dim=768, output_dim=768)
#         #self.promptlearner = PromptLearner(dim_text=768,dim_image=768,dim_out=768)
#         self.promptlearner = PromptLearner()
#     def forward(self, x):
#
#         # 将输入通过图像编码器的第一个卷积层
#         x = self.image_encoder.conv1(x)
#         # 将卷积层的输出重塑为 [batch_size, channels, num_patches]
#         x = x.reshape(x.shape[0], x.shape[1], -1)
#         # 在重塑为将卷积层的输出重塑为 [batch_size, num_patches, channels]
#         x = x.permute(0, 2, 1)
#         # 在第1维度上拼接别嵌入（class embedding）和卷积层的输出
#         x = torch.cat(
#             [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0],
#                     1, x.shape[-1], dtype=x.dtype, device=x.device),x], dim=1)
#         # 将位置嵌入（positional,embedding）加到输入上。
#         x = x + self.image_encoder.positional_embedding.to(x.dtype)
#         # 丢弃一些patch
#         x = self.image_encoder.patch_dropout(x)
#         x = self.image_encoder.ln_pre(x)
#
#         # 将张量的维度重新排列为 [num_patches, batch_size, channels]
#         x = x.permute(1, 0, 2)
#
#
#         attn_out = []
#         seg_patch_tokens = []
#         det_patch_tokens = []
#
#         for i in range(24):
#
#            # 保留一层的输出注意力，来获得其维度
# ############################################################################################
#             if i + 1 == 12:
#                 x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
#                 attn_out.append(attn)
# #############################################################################################
#
#             # 这里是先经过vit
#             else:
#                 x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
#
#             # 四个patch特征,放到6，12,18，24层的vit添加适配器
#             if (i + 1) in self.features:
#
#                 # 这里的seg_adapt_med,det_adapt_med的维度是(L,B,C)
#                 seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i+1)](x)
#                 det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x)
#
#                 x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out
#                 # 这里的seg_patch_tokens,det_patch_tokens内元素的维度是(L,B,C)
#                 seg_patch_tokens.append(seg_adapt_med)
#                 det_patch_tokens.append(det_adapt_med)
#
# # 这里感觉没有用
# #########################################################################################
#         # # 获得特征的图的维度：（B，C，L）
#         # # attn_out 只保留了12层的注意力图
#         # B, C, L = attn_out[0].shape
#         # # 计算
#         # H = int(math.sqrt(L-1))
#         # out_attn = torch.zeros([H, H]).to('cuda')
#         #
#         # for i in range(len(attn)):
#         #     out_attn = out_attn + attn_out[i][0, 0, 1:].view(H, H)
#
# ##########################################################################################
#
#         # 将张量的维度重新排列为[batch_size, num_patches, channels]
#         x = x.permute(1, 0, 2)
#
#         # 将中间级的维度重新排列为[batch_size, num_patches, channels] BLC
#         seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
#         det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]
#
#         pooled, tokens = self.image_encoder._global_pool(x)
#         pooled = self.image_encoder.ln_post(pooled)
#         if self.image_encoder.proj is not None:
#             pooled = pooled @ self.image_encoder.proj
#         # 获取图像特征
#         image_features = det_patch_tokens[-1]  # 假设最后一层的特征是我们需要的
#         prompt_output = self.promptlearner(self.text_features, image_features)
#         # 返回四个中间级特征
#         return pooled, seg_patch_tokens, det_patch_tokens, prompt_output
#
#
# if __name__ == '__main__':
#         # 设置设备，优先使用GPU
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#
#         # 创建基础的ViT模型，这里假设你有一个 create_model 函数用于创建模型
#         model = create_model('ViT-L-14-336', 240).to(device)
#
#         # 将基础模型传入 CLIP_Inplanted 以进行扩展
#         model1 = CLIP_Inplanted(model, [6, 12, 18, 24]).to(device)
#
#         # 打印模型结构摘要
#         from torchsummary import summary
#
#         print(summary(model1, (3, 336, 336)))

