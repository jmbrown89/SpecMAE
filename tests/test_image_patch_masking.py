from __future__ import annotations

import torch

from specmae.masking.image import apply_random_patch_mask


def test_image_patch_masking_preserves_shape_and_range() -> None:
    images = torch.rand(8, 1, 28, 28)
    corrupted, keep_mask = apply_random_patch_mask(images=images, mask_ratio=0.6, patch_size=4)

    assert corrupted.shape == images.shape
    assert keep_mask.shape == (8, 1, 28, 28)
    assert corrupted.min().item() >= 0.0
    assert corrupted.max().item() <= 1.0


def test_image_patch_masking_has_masked_pixels() -> None:
    images = torch.ones(4, 1, 28, 28)
    corrupted, keep_mask = apply_random_patch_mask(images=images, mask_ratio=0.6, patch_size=4)

    assert torch.count_nonzero(keep_mask == 0).item() > 0
    assert torch.count_nonzero(corrupted == 0).item() > 0
