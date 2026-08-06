from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from specmae.data.medmnist import MedMNISTConfig, load_medmnist_dataset, normalize_sample


def test_normalize_sample_structure() -> None:
	image = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
	label = 1
	image_tensor, label_tensor = normalize_sample((image, label), image_size=28)

	assert isinstance(image_tensor, torch.Tensor)
	assert isinstance(label_tensor, torch.Tensor)
	assert image_tensor.shape == (1, 28, 28)
	assert label_tensor.shape == (1,)


@pytest.mark.skipif(
	os.environ.get("SPECMAE_RUN_DATA_TESTS") != "1",
	reason="Set SPECMAE_RUN_DATA_TESTS=1 to run integration test that may download MedMNIST.",
)
def test_medmnist_dataset_loading_integration() -> None:
	dataset = load_medmnist_dataset(
		MedMNISTConfig(
			dataset_name="pathmnist",
			split="train",
			root="data/raw",
			download=True,
			image_size=28,
			limit_samples=2,
		)
	)
	sample = dataset[0]
	assert isinstance(sample, tuple)
	assert len(sample) == 2

