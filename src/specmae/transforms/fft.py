"""Fourier transform backend for SpecMAE."""

from __future__ import annotations

import torch

from specmae.transforms.base import Transform2D


class FourierTransform2D(Transform2D):
    """2D orthonormal FFT transform for real-valued image tensors."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got shape {tuple(images.shape)}")
        return torch.fft.fft2(images, dim=(-2, -1), norm="ortho")

    def inverse(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got shape {tuple(coefficients.shape)}")
        return torch.fft.ifft2(coefficients, dim=(-2, -1), norm="ortho").real
