"""Abstract interfaces for image<->coefficient transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Transform2D(ABC):
    """Transform contract for 2D image tensors shaped [B, C, H, W]."""

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map real-valued images to coefficient space."""

    @abstractmethod
    def inverse(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Map coefficients back to real-valued image space."""
