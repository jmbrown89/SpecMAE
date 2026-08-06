"""Reconstruction model entrypoint for SpecMAE."""

from __future__ import annotations

import torch
from torch import nn

from specmae.models.resunet import ResUNetAutoencoder


class TinyAutoencoder(nn.Module):
	"""Compatibility wrapper around the lightweight ResUNet backbone."""

	def __init__(self, in_channels: int, hidden_dim: int = 32, latent_dim: int = 64) -> None:
		super().__init__()
		_ = latent_dim
		self.backbone = ResUNetAutoencoder(in_channels=in_channels, base_channels=hidden_dim)
		self.encoder = self.backbone.encoder

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.backbone(x)

