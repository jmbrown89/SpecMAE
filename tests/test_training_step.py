from __future__ import annotations

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from specmae.models.mae import TinyAutoencoder
from specmae.spectral.mask import SpectralMaskConfig
from specmae.training.train import train_one_epoch


def test_single_training_epoch_runs() -> None:
	images = torch.rand(16, 1, 28, 28)
	labels = torch.zeros(16, 1, dtype=torch.long)
	dataset = TensorDataset(images, labels)
	loader = DataLoader(dataset, batch_size=8, shuffle=False)

	model = TinyAutoencoder(in_channels=1)
	optimizer = Adam(model.parameters(), lr=1e-3)
	loss = train_one_epoch(
		model=model,
		dataloader=loader,
		optimizer=optimizer,
		device=torch.device("cpu"),
		mask_config=SpectralMaskConfig(policy="high_freq", mask_ratio=0.6),
	)

	assert isinstance(loss, float)
	assert loss >= 0.0

