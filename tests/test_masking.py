from __future__ import annotations

import torch

from specmae.spectral.fft import fft2_image
from specmae.spectral.mask import SpectralMaskConfig, apply_keep_mask, make_keep_mask


def test_make_keep_mask_shape_and_dtype() -> None:
	mask = make_keep_mask(28, 28, SpectralMaskConfig(policy="high_freq", mask_ratio=0.6))
	assert mask.shape == (28, 28)
	assert mask.dtype == torch.bool


def test_apply_keep_mask_preserves_tensor_shape_dtype() -> None:
	images = torch.rand(2, 1, 28, 28)
	spectrum = fft2_image(images)
	keep = make_keep_mask(28, 28, SpectralMaskConfig(policy="random", mask_ratio=0.5))
	masked = apply_keep_mask(spectrum, keep)
	assert masked.shape == spectrum.shape
	assert masked.dtype == spectrum.dtype


def test_mid_freq_mask_matches_requested_ratio() -> None:
	requested_mask_ratio = 0.6
	keep = make_keep_mask(28, 28, SpectralMaskConfig(policy="mid_freq", mask_ratio=requested_mask_ratio))
	actual_mask_ratio = 1.0 - keep.float().mean().item()
	assert abs(actual_mask_ratio - requested_mask_ratio) < 0.02


def test_random_mask_is_conjugate_symmetric_in_unshifted_layout() -> None:
	keep = make_keep_mask(
		28,
		28,
		SpectralMaskConfig(policy="random", mask_ratio=0.5),
		centered_output=False,
	)
	for y in range(28):
		for x in range(28):
			y2 = (-y) % 28
			x2 = (-x) % 28
			assert bool(keep[y, x]) == bool(keep[y2, x2])

