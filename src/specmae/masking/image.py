"""Image-space masking utilities for MAE-style baselines."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_random_patch_mask(
    images: torch.Tensor,
    mask_ratio: float,
    patch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random patch masking in image space.

    Returns:
        corrupted_images: Masked image tensor with same shape as input.
        keep_mask: [B, 1, H, W] mask where 1 means kept pixel.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got shape {tuple(images.shape)}")
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError(f"Expected mask_ratio in [0, 1), got {mask_ratio}")
    if patch_size <= 0:
        raise ValueError(f"Expected patch_size > 0, got {patch_size}")

    batch_size, _, height, width = images.shape
    grid_h = (height + patch_size - 1) // patch_size
    grid_w = (width + patch_size - 1) // patch_size

    keep_grid = torch.rand(batch_size, 1, grid_h, grid_w, device=images.device, dtype=images.dtype) >= mask_ratio
    keep_mask = F.interpolate(keep_grid.float(), size=(height, width), mode="nearest").to(images.dtype)
    corrupted = images * keep_mask
    return corrupted, keep_mask
