from __future__ import annotations

import torch

from specmae.spectral.fft import fft2_image, ifft2_image


def test_fft_ifft_roundtrip_close() -> None:
	images = torch.rand(4, 1, 28, 28)
	spectrum = fft2_image(images)
	reconstructed = ifft2_image(spectrum)
	assert torch.allclose(images, reconstructed, atol=1e-5)

