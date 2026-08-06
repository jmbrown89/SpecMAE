"""Masking policies and schedulers for SpecMAE."""

from specmae.spectral.mask import SpectralMaskConfig, apply_keep_mask, make_keep_mask, radial_frequency_grid
from specmae.masking.curriculum import CurriculumStage, FixedMaskScheduler, LinearMaskScheduler, SpectralCurriculumScheduler, evenly_split_stage_ends
from specmae.masking.image import apply_random_patch_mask

__all__ = [
    "SpectralMaskConfig",
    "apply_keep_mask",
    "make_keep_mask",
    "radial_frequency_grid",
    "CurriculumStage",
    "FixedMaskScheduler",
    "LinearMaskScheduler",
    "SpectralCurriculumScheduler",
    "evenly_split_stage_ends",
    "apply_random_patch_mask",
]
