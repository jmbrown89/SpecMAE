from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from specmae.utils.seed import make_torch_generator, set_seed


def _batch_order(loader: DataLoader) -> list[int]:
    order: list[int] = []
    for batch in loader:
        features, _ = batch
        order.extend(features.squeeze(-1).tolist())
    return [int(x) for x in order]


def test_set_seed_reproducible_torch_and_numpy() -> None:
    set_seed(123)
    torch_a = torch.rand(8)
    np_a = np.random.rand(8)

    set_seed(123)
    torch_b = torch.rand(8)
    np_b = np.random.rand(8)

    assert torch.allclose(torch_a, torch_b)
    assert np.allclose(np_a, np_b)


def test_dataloader_shuffle_reproducible_with_generator() -> None:
    features = torch.arange(24).unsqueeze(-1).float()
    labels = torch.zeros(24, 1)
    dataset = TensorDataset(features, labels)

    loader_a = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        generator=make_torch_generator(99),
    )
    loader_b = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        generator=make_torch_generator(99),
    )

    assert _batch_order(loader_a) == _batch_order(loader_b)
