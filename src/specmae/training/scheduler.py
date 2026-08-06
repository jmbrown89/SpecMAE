"""Phase 2 curriculum scheduling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from specmae.spectral.mask import SpectralMaskConfig


@dataclass(frozen=True)
class EpochStage:
	"""Mask ratio applied until and including a specific epoch index."""

	end_epoch: int
	mask_ratio: float


class EpochStageMaskScheduler:
	"""Resolve mask ratio by epoch based on ordered stages."""

	def __init__(self, stages: List[EpochStage], default_mask_ratio: float) -> None:
		self._stages = sorted(stages, key=lambda stage: stage.end_epoch)
		self._default = default_mask_ratio

	def mask_ratio_at(self, epoch: int) -> float:
		for stage in self._stages:
			if epoch <= stage.end_epoch:
				return stage.mask_ratio
		return self._default


@dataclass(frozen=True)
class CurriculumStage:
	"""One curriculum stage for policy and ratio progression.

	Exactly one of end_epoch or end_step should be used depending on the scheduler mode.
	"""

	policy: str
	mask_ratio_start: float
	mask_ratio_end: float
	end_epoch: int | None = None
	end_step: int | None = None


def evenly_split_stage_ends(total_count: int, stage_count: int) -> List[int]:
	"""Return inclusive end indices that evenly split [0, total_count - 1]."""
	if total_count <= 0:
		raise ValueError("total_count must be > 0")
	if stage_count <= 0:
		raise ValueError("stage_count must be > 0")

	ends: List[int] = []
	for idx in range(1, stage_count + 1):
		end = int(round(idx * total_count / stage_count)) - 1
		ends.append(min(total_count - 1, max(0, end)))
	ends[-1] = total_count - 1
	return ends


class SpectralCurriculumScheduler:
	"""Resolve spectral mask config by epoch or global step.

	Supports:
	- fixed stage boundaries by epoch or step,
	- progressive policy activation (e.g. high -> mid -> low),
	- optional gradual mask-ratio increase inside each stage.
	"""

	def __init__(
		self,
		stages: Sequence[CurriculumStage],
		mode: str = "epoch",
		default_policy: str = "high_frequency_first",
		default_mask_ratio: float = 0.6,
	) -> None:
		if mode not in {"epoch", "step"}:
			raise ValueError("mode must be one of {'epoch', 'step'}")
		if not stages:
			raise ValueError("At least one curriculum stage is required")

		self._mode = mode
		self._default_policy = default_policy
		self._default_mask_ratio = default_mask_ratio
		self._stages = list(stages)

		if mode == "epoch":
			self._stages = sorted(self._stages, key=lambda stage: stage.end_epoch if stage.end_epoch is not None else -1)
			if any(stage.end_epoch is None for stage in self._stages):
				raise ValueError("All stages must define end_epoch when mode='epoch'")
		else:
			self._stages = sorted(self._stages, key=lambda stage: stage.end_step if stage.end_step is not None else -1)
			if any(stage.end_step is None for stage in self._stages):
				raise ValueError("All stages must define end_step when mode='step'")

	@classmethod
	def high_mid_low_by_epoch(
		cls,
		total_epochs: int,
		mask_ratio_start: float,
		mask_ratio_end: float,
	) -> "SpectralCurriculumScheduler":
		"""Build a default high->mid->low curriculum over epochs."""
		stage_ends = evenly_split_stage_ends(total_count=total_epochs, stage_count=3)
		range_size = mask_ratio_end - mask_ratio_start
		mid_1 = mask_ratio_start + (range_size / 3.0)
		mid_2 = mask_ratio_start + (2.0 * range_size / 3.0)

		stages = [
			CurriculumStage(policy="high_frequency_first", mask_ratio_start=mask_ratio_start, mask_ratio_end=mid_1, end_epoch=stage_ends[0]),
			CurriculumStage(policy="radial", mask_ratio_start=mid_1, mask_ratio_end=mid_2, end_epoch=stage_ends[1]),
			CurriculumStage(policy="low_frequency_first", mask_ratio_start=mid_2, mask_ratio_end=mask_ratio_end, end_epoch=stage_ends[2]),
		]
		return cls(stages=stages, mode="epoch", default_policy="low_frequency_first", default_mask_ratio=mask_ratio_end)

	def _stage_key(self, stage: CurriculumStage) -> int:
		if self._mode == "epoch":
			assert stage.end_epoch is not None
			return stage.end_epoch
		assert stage.end_step is not None
		return stage.end_step

	def _interpolate_ratio(self, stage: CurriculumStage, prev_end: int, current: int) -> float:
		stage_end = self._stage_key(stage)
		stage_start = prev_end + 1
		if stage_end <= stage_start:
			return stage.mask_ratio_end

		position = min(max(current, stage_start), stage_end)
		alpha = (position - stage_start) / float(stage_end - stage_start)
		return stage.mask_ratio_start + alpha * (stage.mask_ratio_end - stage.mask_ratio_start)

	def config_at(self, epoch: int, global_step: int) -> SpectralMaskConfig:
		"""Return the curriculum-derived mask config for current epoch/step."""
		current = epoch if self._mode == "epoch" else global_step
		prev_end = -1
		for stage in self._stages:
			stage_end = self._stage_key(stage)
			if current <= stage_end:
				ratio = self._interpolate_ratio(stage=stage, prev_end=prev_end, current=current)
				ratio = min(max(ratio, 0.0), 0.999)
				return SpectralMaskConfig(policy=stage.policy, mask_ratio=ratio)
			prev_end = stage_end

		return SpectralMaskConfig(policy=self._default_policy, mask_ratio=self._default_mask_ratio)


class FixedMaskScheduler:
	"""Return a constant mask policy and ratio."""

	def __init__(self, policy: str, mask_ratio: float) -> None:
		self._policy = policy
		self._mask_ratio = mask_ratio

	def config_at(self, epoch: int, global_step: int) -> SpectralMaskConfig:
		_ = epoch
		_ = global_step
		return SpectralMaskConfig(policy=self._policy, mask_ratio=self._mask_ratio)


class LinearMaskScheduler:
	"""Linearly increase mask ratio over total epochs or total steps."""

	def __init__(
		self,
		policy: str,
		mask_ratio_start: float,
		mask_ratio_end: float,
		total_count: int,
		mode: str = "epoch",
	) -> None:
		if mode not in {"epoch", "step"}:
			raise ValueError("mode must be one of {'epoch', 'step'}")
		if total_count <= 0:
			raise ValueError("total_count must be > 0")
		self._policy = policy
		self._start = mask_ratio_start
		self._end = mask_ratio_end
		self._total = total_count
		self._mode = mode

	def config_at(self, epoch: int, global_step: int) -> SpectralMaskConfig:
		pos = epoch if self._mode == "epoch" else global_step
		if self._total <= 1:
			ratio = self._end
		else:
			alpha = min(max(pos, 0), self._total - 1) / float(self._total - 1)
			ratio = self._start + alpha * (self._end - self._start)
		ratio = min(max(ratio, 0.0), 0.999)
		return SpectralMaskConfig(policy=self._policy, mask_ratio=ratio)


