import torch
import torch.nn.functional as F
import torch.nn as nn
from ..modules.ops import video_to_image


class HaarWaveletTransform2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('aa', torch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('ad', torch.tensor([[1, 1], [-1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('da', torch.tensor([[1, -1], [1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('dd', torch.tensor([[1, -1], [-1, 1]]).view(1, 1, 2, 2) / 2)

    @video_to_image
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * c, 1, h, w)
        low_low = F.conv2d(x, self.aa, stride=2).reshape(b, c, h // 2, w // 2)
        low_high = F.conv2d(x, self.ad, stride=2).reshape(b, c, h // 2, w // 2)
        high_low = F.conv2d(x, self.da, stride=2).reshape(b, c, h // 2, w // 2)
        high_high = F.conv2d(x, self.dd, stride=2).reshape(b, c, h // 2, w // 2)
        coeffs = torch.cat([low_low, low_high, high_low, high_high], dim=1)
        return coeffs

class InverseHaarWaveletTransform2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('aa', torch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('ad', torch.tensor([[1, 1], [-1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('da', torch.tensor([[1, -1], [1, -1]]).view(1, 1, 2, 2) / 2)
        self.register_buffer('dd', torch.tensor([[1, -1], [-1, 1]]).view(1, 1, 2, 2) / 2)

    @video_to_image
    def forward(self, coeffs):
        low_low, low_high, high_low, high_high = coeffs.chunk(4, dim=1)
        b, c, height_half, width_half = low_low.shape
        height = height_half * 2
        width = width_half * 2

        low_low = F.conv_transpose2d(
            low_low.reshape(b * c, 1, height_half, width_half), self.aa, stride=2
        )
        low_high = F.conv_transpose2d(
            low_high.reshape(b * c, 1, height_half, width_half), self.ad, stride=2
        )
        high_low = F.conv_transpose2d(
            high_low.reshape(b * c, 1, height_half, width_half), self.da, stride=2
        )
        high_high = F.conv_transpose2d(
            high_high.reshape(b * c, 1, height_half, width_half), self.dd, stride=2
        )

        return (low_low + low_high + high_low + high_high).reshape(b, c, height, width)