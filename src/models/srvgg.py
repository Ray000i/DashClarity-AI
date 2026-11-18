"""
SRVGGNetCompact Architecture
Compact VGG-style Super-Resolution Network used by Real-ESRGAN (e.g. realesr-general-x4v3).
Fast, tiled-friendly, and optimal for real-time video restoration on 8 GB VRAM.
Reference: Wang et al., "Real-ESRGAN: Training Real-World Image Super-Resolution with Pure Synthetic Data", ICCVW 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    """
    Compact VGG-style network for super-resolution.
    realesr-general-x4v3 uses num_feat=64, num_conv=32, upscale=4.
    """
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
        act_type: str = 'prelu'
    ):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # First convolution
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # Trunk convolutions (num_conv layers total)
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # Upsampling and output convolution
        self.body.append(nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body[:-1]:
            out = layer(out)
        out = self.body[-1](out)
        out = self.upsampler(out)
        
        # Add residual base interpolation
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out = out + base
        return out
