"""Spectral masking policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SpectralMaskConfig:
	"""Mask policy configuration.

	policy:
		- "high_frequency_first": mask high frequencies first (keep lower frequencies).
		- "low_frequency_first": mask low frequencies first (keep higher frequencies).
		- "radial": keep middle-band coefficients around radial mid-frequencies.
		- "random": randomly keep coefficients.
	mask_ratio:
		Fraction of coefficients to mask in [0, 1).
	"""

	policy: str = "high_frequency_first"
	mask_ratio: float = 0.6


_POLICY_ALIASES: Dict[str, str] = {
	"high_freq": "high_frequency_first",
	"low_freq": "low_frequency_first",
	"mid_freq": "radial",
	"high_frequency_first": "high_frequency_first",
	"low_frequency_first": "low_frequency_first",
	"radial": "radial",
	"random": "random",
}


def _canonical_policy(policy: str) -> str:
	if policy not in _POLICY_ALIASES:
		raise ValueError(f"Unknown mask policy: {policy}")
	return _POLICY_ALIASES[policy]


def radial_frequency_grid(
	height: int,
	width: int,
	device: Optional[torch.device] = None,
	centered: bool = True,
) -> torch.Tensor:
	"""Return normalized radial distances for frequency bins.

	When centered=False, the grid follows raw FFT indexing and aligns with conjugate pairs.
	"""
	fy = torch.fft.fftfreq(height, d=1.0, device=device)
	fx = torch.fft.fftfreq(width, d=1.0, device=device)
	yy, xx = torch.meshgrid(fy, fx, indexing="ij")
	radius = torch.sqrt(xx**2 + yy**2)
	radius = radius / radius.max().clamp_min(1e-8)
	if centered:
		radius = torch.fft.fftshift(radius, dim=(-2, -1))
	return radius


def _pair_groups(height: int, width: int) -> List[List[Tuple[int, int]]]:
	"""Build groups of conjugate index pairs in raw FFT indexing."""
	seen = set()
	groups: List[List[Tuple[int, int]]] = []
	for y in range(height):
		for x in range(width):
			if (y, x) in seen:
				continue
			y2 = (-y) % height
			x2 = (-x) % width
			if y == y2 and x == x2:
				group = [(y, x)]
			else:
				group = [(y, x), (y2, x2)]
			for coord in group:
				seen.add(coord)
			groups.append(group)
	return groups


def random_keep_mask(
	height: int,
	width: int,
	keep_ratio: float,
	device: Optional[torch.device] = None,
	generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
	"""Build conjugate-symmetric random keep mask of shape [H, W]."""
	return make_keep_mask(
		height=height,
		width=width,
		config=SpectralMaskConfig(policy="random", mask_ratio=1.0 - keep_ratio),
		device=device,
		generator=generator,
		centered_output=True,
	)


def make_keep_mask(
	height: int,
	width: int,
	config: SpectralMaskConfig,
	device: Optional[torch.device] = None,
	generator: Optional[torch.Generator] = None,
	centered_output: bool = True,
) -> torch.Tensor:
	"""Create a conjugate-symmetric boolean keep mask [H, W] from config."""
	if not 0.0 <= config.mask_ratio < 1.0:
		raise ValueError("mask_ratio must be in [0, 1)")

	policy = _canonical_policy(config.policy)
	keep_ratio = 1.0 - config.mask_ratio
	total = height * width
	keep_count = int(round(keep_ratio * total))
	keep_count = max(0, min(total, keep_count))

	if keep_count == 0:
		mask = torch.zeros((height, width), dtype=torch.bool, device=device)
		return torch.fft.fftshift(mask, dim=(-2, -1)) if centered_output else mask
	if keep_count == total:
		mask = torch.ones((height, width), dtype=torch.bool, device=device)
		return torch.fft.fftshift(mask, dim=(-2, -1)) if centered_output else mask

	radius = radial_frequency_grid(height, width, device=device, centered=False)
	groups = _pair_groups(height, width)

	def group_score(group: Sequence[Tuple[int, int]]) -> float:
		if policy == "random":
			return float(torch.rand((), generator=generator).item())
		vals = [float(radius[y, x].item()) for (y, x) in group]
		avg_r = sum(vals) / len(vals)
		if policy == "high_frequency_first":
			return -avg_r
		if policy == "low_frequency_first":
			return avg_r
		# radial: keep center of spectral radius distribution.
		return -abs(avg_r - 0.5)

	scored = []
	for group in groups:
		scored.append((group_score(group), group))
	scored.sort(key=lambda item: item[0], reverse=True)

	keep_mask = torch.zeros((height, width), dtype=torch.bool, device=device)
	kept = 0
	for _, group in scored:
		group_size = len(group)
		if kept >= keep_count:
			break
		if kept + group_size <= keep_count:
			select = True
		else:
			select = abs((kept + group_size) - keep_count) <= abs(kept - keep_count)
		if not select:
			continue
		for y, x in group:
			keep_mask[y, x] = True
		kept += group_size

	if centered_output:
		return torch.fft.fftshift(keep_mask, dim=(-2, -1))
	return keep_mask


def apply_keep_mask(spectrum: torch.Tensor, keep_mask_hw: torch.Tensor) -> torch.Tensor:
	"""Apply keep mask over [H, W] to a spectrum [B, C, H, W]."""
	if spectrum.ndim != 4:
		raise ValueError(f"Expected [B, C, H, W], got shape {tuple(spectrum.shape)}")
	if keep_mask_hw.ndim != 2:
		raise ValueError(f"Expected [H, W] mask, got shape {tuple(keep_mask_hw.shape)}")
	mask = keep_mask_hw.to(dtype=spectrum.dtype, device=spectrum.device).unsqueeze(0).unsqueeze(0)
	return spectrum * mask

