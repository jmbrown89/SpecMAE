from __future__ import annotations

from specmae.training.scheduler import CurriculumStage, FixedMaskScheduler, LinearMaskScheduler, SpectralCurriculumScheduler, evenly_split_stage_ends


def test_evenly_split_stage_ends() -> None:
	assert evenly_split_stage_ends(total_count=9, stage_count=3) == [2, 5, 8]


def test_high_mid_low_progression_by_epoch() -> None:
	scheduler = SpectralCurriculumScheduler.high_mid_low_by_epoch(
		total_epochs=9,
		mask_ratio_start=0.3,
		mask_ratio_end=0.9,
	)

	cfg0 = scheduler.config_at(epoch=0, global_step=0)
	cfg4 = scheduler.config_at(epoch=4, global_step=0)
	cfg8 = scheduler.config_at(epoch=8, global_step=0)

	assert cfg0.policy == "high_frequency_first"
	assert cfg4.policy == "radial"
	assert cfg8.policy == "low_frequency_first"
	assert cfg0.mask_ratio < cfg4.mask_ratio < cfg8.mask_ratio


def test_step_curriculum_progression() -> None:
	stages = [
		CurriculumStage(policy="high_frequency_first", mask_ratio_start=0.3, mask_ratio_end=0.4, end_step=2),
		CurriculumStage(policy="radial", mask_ratio_start=0.4, mask_ratio_end=0.5, end_step=5),
		CurriculumStage(policy="low_frequency_first", mask_ratio_start=0.5, mask_ratio_end=0.6, end_step=8),
	]
	scheduler = SpectralCurriculumScheduler(
		stages=stages,
		mode="step",
		default_policy="low_frequency_first",
		default_mask_ratio=0.6,
	)

	cfg1 = scheduler.config_at(epoch=0, global_step=1)
	cfg4 = scheduler.config_at(epoch=0, global_step=4)
	cfg7 = scheduler.config_at(epoch=0, global_step=7)

	assert cfg1.policy == "high_frequency_first"
	assert cfg4.policy == "radial"
	assert cfg7.policy == "low_frequency_first"
	assert 0.3 <= cfg1.mask_ratio <= 0.4
	assert 0.4 <= cfg4.mask_ratio <= 0.5
	assert 0.5 <= cfg7.mask_ratio <= 0.6


def test_fixed_scheduler() -> None:
	scheduler = FixedMaskScheduler(policy="random", mask_ratio=0.55)
	cfg = scheduler.config_at(epoch=10, global_step=200)
	assert cfg.policy == "random"
	assert cfg.mask_ratio == 0.55


def test_linear_scheduler() -> None:
	scheduler = LinearMaskScheduler(
		policy="high_frequency_first",
		mask_ratio_start=0.2,
		mask_ratio_end=0.8,
		total_count=5,
		mode="epoch",
	)
	cfg0 = scheduler.config_at(epoch=0, global_step=0)
	cfg2 = scheduler.config_at(epoch=2, global_step=0)
	cfg4 = scheduler.config_at(epoch=4, global_step=0)
	assert cfg0.mask_ratio < cfg2.mask_ratio < cfg4.mask_ratio
