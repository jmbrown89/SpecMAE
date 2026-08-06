"""Random seeding utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
	"""Set deterministic seeds for Python, NumPy, and PyTorch."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	if deterministic:
		# This is required by some CUDA backends for deterministic kernels.
		os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.use_deterministic_algorithms(True, warn_only=True)
	else:
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True
		torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
	"""Seed a dataloader worker using the worker's PyTorch-derived seed."""
	del worker_id
	worker_seed = torch.initial_seed() % (2**32)
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def make_torch_generator(seed: int, device: str = "cpu") -> torch.Generator:
	"""Create a manually seeded PyTorch generator."""
	generator = torch.Generator(device=device)
	generator.manual_seed(seed)
	return generator

