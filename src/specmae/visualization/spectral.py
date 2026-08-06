"""Utilities for visualizing spectral masking behavior."""

from __future__ import annotations

from typing import Dict

import torch


def build_spectral_debug_tensors(
    original_images: torch.Tensor,
    corrupted_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    keep_mask_centered: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return key tensors for FFT-domain debugging visualizations.

    Keys include original/corrupted/reconstructed FFT log-magnitudes in centered
    coordinates, plus the centered binary keep mask.
    """
    if original_images.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got shape {tuple(original_images.shape)}")

    original_fft = torch.fft.fftshift(torch.fft.fft2(original_images, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    corrupted_fft = torch.fft.fftshift(torch.fft.fft2(corrupted_images, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    recon_fft = torch.fft.fftshift(torch.fft.fft2(reconstructed_images, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

    return {
        "original_image": original_images,
        "corrupted_image": corrupted_images,
        "reconstructed_image": reconstructed_images,
        "original_fft_log_mag": torch.log1p(torch.abs(original_fft)),
        "corrupted_fft_log_mag": torch.log1p(torch.abs(corrupted_fft)),
        "reconstructed_fft_log_mag": torch.log1p(torch.abs(recon_fft)),
        "keep_mask_centered": keep_mask_centered,
    }
