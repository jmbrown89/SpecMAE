"""2D FFT utilities for real-valued image tensors."""

from __future__ import annotations

import torch

from specmae.transforms.fft import FourierTransform2D


_FFT = FourierTransform2D()


def fft2_image(images: torch.Tensor) -> torch.Tensor:
	"""Apply 2D FFT via the transform backend."""
	return _FFT.forward(images)


def ifft2_image(spectrum: torch.Tensor) -> torch.Tensor:
	"""Apply inverse 2D FFT via the transform backend."""
	return _FFT.inverse(spectrum)

