"""Loss functions for reconstruction training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
	"""Configuration for image-space reconstruction losses."""

	kind: str = "mse"
	mse_weight: float = 1.0
	l1_weight: float = 1.0


def compute_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, config: LossConfig) -> torch.Tensor:
	"""Compute configured reconstruction loss.

	This API intentionally isolates image-space losses so frequency-domain terms
	can be added later without changing training loops.
	"""
	kind = config.kind
	if kind == "mse":
		return F.mse_loss(prediction, target)
	if kind == "l1":
		return F.l1_loss(prediction, target)
	if kind == "combined":
		mse = F.mse_loss(prediction, target)
		l1 = F.l1_loss(prediction, target)
		return (config.mse_weight * mse) + (config.l1_weight * l1)
	raise ValueError(f"Unknown loss kind: {kind}")


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	"""Backward-compatible default reconstruction loss (MSE)."""
	return compute_reconstruction_loss(prediction, target, LossConfig(kind="mse"))

