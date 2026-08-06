"""Image transform helpers for MedMNIST-like inputs."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F


def to_chw_float_tensor(image: Any, image_size: Optional[int] = None) -> torch.Tensor:
	"""Convert input image to normalized CHW float tensor in [0, 1]."""
	if isinstance(image, torch.Tensor):
		tensor = image.clone().detach()
	else:
		array = np.array(image)
		if array.ndim == 2:
			array = np.expand_dims(array, axis=-1)
		tensor = torch.from_numpy(array)

	if tensor.ndim == 2:
		tensor = tensor.unsqueeze(0)
	elif tensor.ndim == 3 and tensor.shape[0] not in (1, 3):
		tensor = tensor.permute(2, 0, 1)

	tensor = tensor.float()
	if tensor.max() > 1.0:
		tensor = tensor / 255.0

	if image_size is not None and (tensor.shape[-2] != image_size or tensor.shape[-1] != image_size):
		tensor = F.interpolate(
			tensor.unsqueeze(0),
			size=(image_size, image_size),
			mode="bilinear",
			align_corners=False,
		).squeeze(0)
	return tensor


class MedMNISTTransform:
	"""Callable transform for MedMNIST dataset outputs."""

	def __init__(self, image_size: Optional[int] = None) -> None:
		self.image_size = image_size

	def __call__(self, image: Any) -> torch.Tensor:
		return to_chw_float_tensor(image, image_size=self.image_size)

