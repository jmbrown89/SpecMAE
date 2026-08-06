"""Metric helpers."""

from __future__ import annotations

import torch


def _safe_float(value: torch.Tensor) -> float:
	return float(value.detach().cpu().item())


def mse_metric(prediction: torch.Tensor, target: torch.Tensor) -> float:
	"""Return scalar mean squared error."""
	return torch.mean((prediction - target) ** 2).item()


def psnr_metric(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
	"""Return PSNR in dB assuming inputs are in [0, data_range]."""
	mse = torch.mean((prediction - target) ** 2)
	mse = torch.clamp(mse, min=1e-12)
	psnr = 10.0 * torch.log10(torch.tensor(data_range * data_range, device=prediction.device) / mse)
	return _safe_float(psnr)


def ssim_metric(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
	"""Return a global SSIM estimate averaged across batch and channels."""
	if prediction.ndim != 4 or target.ndim != 4:
		raise ValueError(f"Expected [B, C, H, W] tensors, got {tuple(prediction.shape)} and {tuple(target.shape)}")

	mu_x = prediction.mean(dim=(-2, -1), keepdim=True)
	mu_y = target.mean(dim=(-2, -1), keepdim=True)

	sigma_x = ((prediction - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
	sigma_y = ((target - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
	sigma_xy = ((prediction - mu_x) * (target - mu_y)).mean(dim=(-2, -1), keepdim=True)

	c1 = (0.01 * data_range) ** 2
	c2 = (0.03 * data_range) ** 2

	numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
	denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
	ssim = numerator / torch.clamp(denominator, min=1e-12)
	return _safe_float(ssim.mean())


def spectral_band_mse_metrics(
	prediction: torch.Tensor,
	target: torch.Tensor,
	low_edge: float = 1.0 / 3.0,
	high_edge: float = 2.0 / 3.0,
) -> dict[str, float]:
	"""Return low/mid/high spectral-domain reconstruction MSE.

	Band edges are radial fractions of the centered 2D frequency radius.
	"""
	if prediction.ndim != 4 or target.ndim != 4:
		raise ValueError(f"Expected [B, C, H, W] tensors, got {tuple(prediction.shape)} and {tuple(target.shape)}")
	if not (0.0 < low_edge < high_edge < 1.0):
		raise ValueError("Expected 0 < low_edge < high_edge < 1")

	_, _, height, width = prediction.shape
	pred_fft = torch.fft.fftshift(torch.fft.fft2(prediction, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
	target_fft = torch.fft.fftshift(torch.fft.fft2(target, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
	diff_power = torch.abs(pred_fft - target_fft) ** 2

	y = torch.arange(height, device=prediction.device, dtype=prediction.dtype) - (height / 2.0)
	x = torch.arange(width, device=prediction.device, dtype=prediction.dtype) - (width / 2.0)
	yy, xx = torch.meshgrid(y, x, indexing="ij")
	radius = torch.sqrt(yy ** 2 + xx ** 2)
	radius = radius / torch.clamp(radius.max(), min=1e-12)

	low_mask = radius <= low_edge
	mid_mask = (radius > low_edge) & (radius <= high_edge)
	high_mask = radius > high_edge

	def _band_mean(mask: torch.Tensor) -> float:
		if not bool(mask.any()):
			return 0.0
		band_vals = diff_power[:, :, mask]
		return _safe_float(band_vals.mean())

	return {
		"low": _band_mean(low_mask),
		"mid": _band_mean(mid_mask),
		"high": _band_mean(high_mask),
	}

