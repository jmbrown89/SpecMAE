"""MedMNIST data loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from specmae.data.transforms import MedMNISTTransform, to_chw_float_tensor
from specmae.utils.seed import make_torch_generator, seed_worker


@dataclass(frozen=True)
class MedMNISTConfig:
	"""Dataset configuration for MedMNIST loading."""

	dataset_name: str = "pathmnist"
	split: str = "train"
	root: str = "data/raw"
	download: bool = True
	image_size: Optional[int] = None
	limit_samples: Optional[int] = None


def normalize_sample(sample: Tuple[Any, Any], image_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Normalize a MedMNIST-like (image, label) sample."""
	image, label = sample
	image_tensor = to_chw_float_tensor(image, image_size=image_size)
	label_tensor = torch.as_tensor(label).long().view(-1)
	return image_tensor, label_tensor


def load_medmnist_dataset(config: MedMNISTConfig, transform: Optional[Any] = None) -> Dataset:
	"""Load a MedMNIST dataset by registry name."""
	try:
		import medmnist  # type: ignore
		from medmnist import INFO  # type: ignore
	except ImportError as exc:
		raise ImportError(
			"medmnist is required. Install dependencies from requirements.txt before running training."
		) from exc

	if config.dataset_name not in INFO:
		available = ", ".join(sorted(INFO.keys()))
		raise ValueError(f"Unknown MedMNIST dataset '{config.dataset_name}'. Available: {available}")

	python_class = INFO[config.dataset_name]["python_class"]
	dataset_class = getattr(medmnist, python_class)
	applied_transform = transform if transform is not None else MedMNISTTransform(image_size=config.image_size)

	dataset = dataset_class(
		split=config.split,
		root=config.root,
		download=config.download,
		transform=applied_transform,
	)

	if config.limit_samples is not None:
		limit = min(config.limit_samples, len(dataset))
		dataset = Subset(dataset, list(range(limit)))
	return dataset


def create_medmnist_dataloader(
	config: MedMNISTConfig,
	batch_size: int,
	shuffle: bool,
	num_workers: int = 0,
	seed: Optional[int] = None,
) -> DataLoader:
	"""Create a DataLoader for MedMNIST with consistent defaults."""
	dataset = load_medmnist_dataset(config=config)
	generator = None
	worker_init_fn = None
	if seed is not None:
		generator = make_torch_generator(seed=seed)
		worker_init_fn = seed_worker

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		generator=generator,
		worker_init_fn=worker_init_fn,
	)

