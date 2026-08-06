"""Metric helpers."""

from __future__ import annotations

import torch


def mse_metric(prediction: torch.Tensor, target: torch.Tensor) -> float:
	"""Return scalar mean squared error."""
	return torch.mean((prediction - target) ** 2).item()

