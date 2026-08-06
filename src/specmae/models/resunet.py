"""Lightweight residual U-Net backbone for reconstruction."""

from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Small residual block with optional projection for channel matching."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.act(out)


class EncoderStage(nn.Module):
    """Residual stage with stride-2 downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.block1 = ResidualBlock(out_channels, out_channels)
        self.block2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.down(x)))
        x = self.block1(x)
        x = self.block2(x)
        return x


class DecoderStage(nn.Module):
    """Upsample + residual fusion with skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class ResUNetEncoder(nn.Module):
    """ResNet-style encoder that can be reused for downstream tasks."""

    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ResidualBlock(c1, c1),
        )
        self.stage1 = EncoderStage(c1, c2)
        self.stage2 = EncoderStage(c2, c3)
        self.stage3 = EncoderStage(c3, c4)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        s0 = self.stem(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return [s0, s1, s2, s3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[-1]


class ResUNetAutoencoder(nn.Module):
    """Residual U-Net for reconstruction of spectrally corrupted images."""

    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = ResUNetEncoder(in_channels=in_channels, base_channels=base_channels)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.dec2 = DecoderStage(in_channels=c4, skip_channels=c3, out_channels=c3)
        self.dec1 = DecoderStage(in_channels=c3, skip_channels=c2, out_channels=c2)
        self.dec0 = DecoderStage(in_channels=c2, skip_channels=c1, out_channels=c1)
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0, s1, s2, s3 = self.encoder.forward_features(x)
        x = self.dec2(s3, s2)
        x = self.dec1(x, s1)
        x = self.dec0(x, s0)
        return self.head(x)
