"""Helpers for spectral corruption and inverse reconstruction."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from specmae.spectral.mask import SpectralMaskConfig, apply_keep_mask, make_keep_mask
from specmae.transforms.base import Transform2D
from specmae.transforms.fft import FourierTransform2D


def corrupt_images_in_spectral_domain(
	images: torch.Tensor,
	mask_config: SpectralMaskConfig,
	generator: Optional[torch.Generator] = None,
	transform: Optional[Transform2D] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Corrupt images by masking Fourier coefficients.

	Returns:
		corrupted_images: Real tensor [B, C, H, W].
		keep_mask_hw: Boolean keep mask [H, W] in centered frequency layout.
	"""
	if images.ndim != 4:
		raise ValueError(f"Expected [B, C, H, W], got shape {tuple(images.shape)}")
	if transform is None:
		transform = FourierTransform2D()

	_, _, height, width = images.shape
	keep_mask_unshifted = make_keep_mask(
		height=height,
		width=width,
		config=mask_config,
		device=images.device,
		generator=generator,
		centered_output=False,
	)
	spectrum = transform.forward(images)
	masked = apply_keep_mask(spectrum, keep_mask_unshifted)
	corrupted = transform.inverse(masked)

	# Return a centered mask for direct visualization in notebooks and logs.
	keep_mask_centered = torch.fft.fftshift(keep_mask_unshifted, dim=(-2, -1))
	return corrupted, keep_mask_centered

