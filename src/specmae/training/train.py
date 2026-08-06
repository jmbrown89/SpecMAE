"""Training entrypoint for spectral corruption reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from specmae.data.medmnist import MedMNISTConfig, create_medmnist_dataloader
from specmae.models.mae import TinyAutoencoder
from specmae.spectral.inverse import corrupt_images_in_spectral_domain
from specmae.spectral.mask import SpectralMaskConfig
from specmae.training.losses import LossConfig, compute_reconstruction_loss
from specmae.training.scheduler import CurriculumStage, FixedMaskScheduler, LinearMaskScheduler, SpectralCurriculumScheduler, evenly_split_stage_ends
from specmae.utils.logging import configure_logging, get_logger
from specmae.utils.metrics import psnr_metric, spectral_band_mse_metrics, ssim_metric
from specmae.utils.seed import set_seed


LOGGER = get_logger(__name__)


def _format_ascii_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
	"""Format rows as a compact ASCII table for terminal logs."""
	str_rows = [[str(cell) for cell in row] for row in rows]
	widths = [len(str(h)) for h in headers]
	for row in str_rows:
		for i, cell in enumerate(row):
			widths[i] = max(widths[i], len(cell))

	def _fmt_row(cells: Sequence[str]) -> str:
		return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

	sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
	lines = [sep, _fmt_row([str(h) for h in headers]), sep]
	lines.extend(_fmt_row(row) for row in str_rows)
	lines.append(sep)
	return "\n".join(lines)


@dataclass(frozen=True)
class TrainingConfig:
	dataset_name: str = "pathmnist"
	data_root: str = "data/raw"
	image_size: int = 28
	batch_size: int = 32
	epochs: int = 1
	learning_rate: float = 1e-3
	mask_ratio: float = 0.6
	mask_policy: str = "high_frequency_first"
	curriculum_mode: str = "none"
	curriculum_mask_ratio_start: float = 0.35
	curriculum_mask_ratio_end: float = 0.8
	loss_kind: str = "mse"
	loss_mse_weight: float = 1.0
	loss_l1_weight: float = 1.0
	artifacts_root: str = "outputs/runs"
	run_name: str = ""
	save_checkpoints: bool = True
	checkpoint_every: int = 1
	save_examples_every: int = 1
	examples_dir_override: str = ""
	early_stopping_enabled: bool = False
	early_stopping_patience: int = 5
	early_stopping_min_delta: float = 0.0
	restore_best_at_end: bool = True
	limit_samples: int = 256
	num_workers: int = 0
	seed: int = 7
	device: str = "cpu"


def _extract_images(batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
	images, _ = batch
	if not isinstance(images, torch.Tensor):
		raise TypeError("Expected tensor images from dataloader")
	return images


def train_one_epoch(
	model: nn.Module,
	dataloader: DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	mask_config: SpectralMaskConfig,
	loss_config: LossConfig | None = None,
) -> float:
	"""Train one epoch and return average loss."""
	model.train()
	total_loss = 0.0
	batches = 0
	if loss_config is None:
		loss_config = LossConfig(kind="mse")

	for batch in dataloader:
		images = _extract_images(batch).to(device)
		corrupted, _ = corrupt_images_in_spectral_domain(images=images, mask_config=mask_config)

		prediction = model(corrupted)
		loss = compute_reconstruction_loss(prediction, images, loss_config)

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		batches += 1

	return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate_reconstruction(
	model: nn.Module,
	dataloader: DataLoader,
	device: torch.device,
	mask_config: SpectralMaskConfig,
	loss_config: LossConfig | None = None,
) -> float:
	"""Evaluate mean reconstruction loss."""
	model.eval()
	total_loss = 0.0
	batches = 0
	if loss_config is None:
		loss_config = LossConfig(kind="mse")

	for batch in dataloader:
		images = _extract_images(batch).to(device)
		corrupted, _ = corrupt_images_in_spectral_domain(images=images, mask_config=mask_config)
		prediction = model(corrupted)
		loss = compute_reconstruction_loss(prediction, images, loss_config)
		total_loss += loss.item()
		batches += 1

	return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate_reconstruction_metrics(
	model: nn.Module,
	dataloader: DataLoader,
	device: torch.device,
	mask_config: SpectralMaskConfig,
	loss_config: LossConfig | None = None,
) -> Dict[str, float]:
	"""Evaluate reconstruction loss plus PSNR/SSIM/spectral-band errors."""
	model.eval()
	if loss_config is None:
		loss_config = LossConfig(kind="mse")

	total_loss = 0.0
	total_psnr = 0.0
	total_ssim = 0.0
	total_spec_low = 0.0
	total_spec_mid = 0.0
	total_spec_high = 0.0
	batches = 0

	for batch in dataloader:
		images = _extract_images(batch).to(device)
		corrupted, _ = corrupt_images_in_spectral_domain(images=images, mask_config=mask_config)
		prediction = model(corrupted)

		loss = compute_reconstruction_loss(prediction, images, loss_config)
		prediction_clamped = prediction.clamp(0.0, 1.0)
		images_clamped = images.clamp(0.0, 1.0)

		spectral = spectral_band_mse_metrics(prediction_clamped, images_clamped)

		total_loss += loss.item()
		total_psnr += psnr_metric(prediction_clamped, images_clamped)
		total_ssim += ssim_metric(prediction_clamped, images_clamped)
		total_spec_low += spectral["low"]
		total_spec_mid += spectral["mid"]
		total_spec_high += spectral["high"]
		batches += 1

	denom = max(batches, 1)
	return {
		"reconstruction_loss": total_loss / denom,
		"psnr": total_psnr / denom,
		"ssim": total_ssim / denom,
		"spectral_mse_low": total_spec_low / denom,
		"spectral_mse_mid": total_spec_mid / denom,
		"spectral_mse_high": total_spec_high / denom,
	}


@torch.no_grad()
def _save_reconstruction_preview(
	model: nn.Module,
	dataloader: DataLoader,
	device: torch.device,
	mask_config: SpectralMaskConfig,
	epoch: int,
	output_dir: str,
) -> None:
	"""Save a small original/corrupted/reconstruction panel for qualitative tracking."""
	try:
		import matplotlib.pyplot as plt
	except Exception as exc:  # pragma: no cover
		LOGGER.warning("Skipping image logging; matplotlib unavailable: %s", exc)
		return

	batch = next(iter(dataloader))
	images = _extract_images(batch).to(device)
	corrupted, _ = corrupt_images_in_spectral_domain(images=images, mask_config=mask_config)
	recon = model(corrupted)

	n = min(4, images.shape[0])
	fig, axes = plt.subplots(n, 3, figsize=(8, 2.0 * n))
	for i in range(n):
		row_axes = axes[i] if n > 1 else axes
		img = images[i].detach().cpu().clamp(0, 1)
		cor = corrupted[i].detach().cpu().clamp(0, 1)
		rec = recon[i].detach().cpu().clamp(0, 1)
		if img.shape[0] == 1:
			row_axes[0].imshow(img[0].numpy(), cmap="gray")
			row_axes[1].imshow(cor[0].numpy(), cmap="gray")
			row_axes[2].imshow(rec[0].numpy(), cmap="gray")
		else:
			row_axes[0].imshow(img.permute(1, 2, 0).numpy())
			row_axes[1].imshow(cor.permute(1, 2, 0).numpy())
			row_axes[2].imshow(rec.permute(1, 2, 0).numpy())
		for ax in row_axes:
			ax.axis("off")

	if n > 0:
		first_row = axes[0] if n > 1 else axes
		first_row[0].set_title("original")
		first_row[1].set_title("corrupted")
		first_row[2].set_title("reconstructed")

	Path(output_dir).mkdir(parents=True, exist_ok=True)
	out_path = Path(output_dir) / f"epoch_{epoch:03d}.png"
	fig.tight_layout()
	fig.savefig(out_path, dpi=120)
	plt.close(fig)
	LOGGER.info("saved_preview=%s", out_path)


def _create_run_artifact_dirs(config: TrainingConfig) -> Dict[str, Path]:
	"""Create a unique run directory tree for artifacts."""
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_name = config.run_name.strip() if config.run_name else f"{config.dataset_name}_{stamp}"
	run_dir = Path(config.artifacts_root) / run_name
	if run_dir.exists():
		run_dir = Path(config.artifacts_root) / f"{run_name}_{stamp}"

	checkpoints_dir = run_dir / "checkpoints"
	metrics_dir = run_dir / "metrics"
	plots_dir = run_dir / "plots"
	if config.examples_dir_override.strip():
		examples_dir = Path(config.examples_dir_override)
	else:
		examples_dir = run_dir / "examples"

	for directory in (run_dir, checkpoints_dir, metrics_dir, plots_dir, examples_dir):
		directory.mkdir(parents=True, exist_ok=True)

	return {
		"run_dir": run_dir,
		"checkpoints_dir": checkpoints_dir,
		"metrics_dir": metrics_dir,
		"plots_dir": plots_dir,
		"examples_dir": examples_dir,
	}


def _save_checkpoint(
	path: Path,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	global_step: int,
	train_loss: float,
	val_loss: float,
	mask_policy: str,
	mask_ratio: float,
	config: TrainingConfig,
) -> None:
	"""Persist model and optimizer state for resume and review."""
	torch.save(
		{
			"epoch": epoch,
			"global_step": global_step,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"train_loss": train_loss,
			"val_loss": val_loss,
			"mask_policy": mask_policy,
			"mask_ratio": mask_ratio,
			"config": asdict(config),
		},
		path,
	)


def _save_history_csv(history: Sequence[Dict[str, Any]], output_path: Path) -> None:
	"""Save epoch-level metrics to CSV."""
	if not history:
		return
	fieldnames = ["epoch", "policy", "mask_ratio", "train_loss", "val_loss"]
	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in history:
			writer.writerow({key: row[key] for key in fieldnames})


def _save_loss_plot(history: Sequence[Dict[str, Any]], output_path: Path) -> None:
	"""Save train/val loss and mask ratio trends as a review plot."""
	if not history:
		return
	try:
		import matplotlib.pyplot as plt
	except Exception as exc:  # pragma: no cover
		LOGGER.warning("Skipping loss plot; matplotlib unavailable: %s", exc)
		return

	epochs = [int(row["epoch"]) for row in history]
	train_losses = [float(row["train_loss"]) for row in history]
	val_losses = [float(row["val_loss"]) for row in history]
	mask_ratios = [float(row["mask_ratio"]) for row in history]

	fig, ax1 = plt.subplots(figsize=(8, 4.5))
	ax1.plot(epochs, train_losses, marker="o", label="train_loss")
	ax1.plot(epochs, val_losses, marker="o", label="val_loss")
	ax1.set_xlabel("epoch")
	ax1.set_ylabel("loss")
	ax1.grid(True, alpha=0.25)

	ax2 = ax1.twinx()
	ax2.plot(epochs, mask_ratios, linestyle="--", color="tab:gray", label="mask_ratio")
	ax2.set_ylabel("mask_ratio")
	ax2.set_ylim(0.0, 1.0)

	handles1, labels1 = ax1.get_legend_handles_labels()
	handles2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
	fig.tight_layout()
	fig.savefig(output_path, dpi=140)
	plt.close(fig)


def _save_run_report(
	run_dir: Path,
	history: Sequence[Dict[str, Any]],
	best_epoch: int,
	best_val_loss: float,
	final_metrics: Dict[str, float],
	stopped_early: bool,
	stop_epoch: int,
	restored_best_at_end: bool,
	config: TrainingConfig,
) -> None:
	"""Write a compact markdown report for fast run review."""
	table_rows = []
	for row in history:
		table_rows.append(
			[
				str(row["epoch"]),
				str(row["policy"]),
				f"{float(row['mask_ratio']):.4f}",
				f"{float(row['train_loss']):.6f}",
				f"{float(row['val_loss']):.6f}",
			]
		)

	report_lines = [
		"# SpecMAE Run Report",
		"",
		f"- Run directory: {run_dir}",
		f"- Dataset: {config.dataset_name}",
		f"- Best epoch: {best_epoch}",
		f"- Best val loss: {best_val_loss:.6f}",
		f"- Stopped early: {stopped_early}",
		f"- Stop epoch: {stop_epoch}",
		f"- Restored best checkpoint at end: {restored_best_at_end}",
		f"- Final val loss: {final_metrics['reconstruction_loss']:.6f}",
		f"- Final PSNR: {final_metrics['psnr']:.4f} dB",
		f"- Final SSIM: {final_metrics['ssim']:.4f}",
		"",
		"## Final Validation Metrics",
		"",
		f"- Reconstruction loss: {final_metrics['reconstruction_loss']:.6f}",
		f"- PSNR: {final_metrics['psnr']:.4f} dB",
		f"- SSIM: {final_metrics['ssim']:.4f}",
		f"- Spectral MSE (low band): {final_metrics['spectral_mse_low']:.6f}",
		f"- Spectral MSE (mid band): {final_metrics['spectral_mse_mid']:.6f}",
		f"- Spectral MSE (high band): {final_metrics['spectral_mse_high']:.6f}",
		"",
		"## Epoch Summary",
		"",
		"```text",
		_format_ascii_table(
			headers=("epoch", "policy", "mask_ratio", "train_loss", "val_loss"),
			rows=table_rows,
		),
		"```",
		"",
		"## Saved Artifacts",
		"",
		"- checkpoints/last.pt",
		"- checkpoints/best_val.pt",
		"- checkpoints/epoch_*.pt",
		"- metrics/history.csv",
		"- metrics/summary.json",
		"- plots/loss_curves.png",
		"- examples/epoch_*.png (or override directory)",
	]

	(run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


def run_training(config: TrainingConfig) -> Dict[str, float]:
	"""Run end-to-end training and return summary metrics."""
	set_seed(config.seed)
	device = torch.device(config.device)
	artifact_dirs = _create_run_artifact_dirs(config)
	run_dir = artifact_dirs["run_dir"]
	checkpoints_dir = artifact_dirs["checkpoints_dir"]
	metrics_dir = artifact_dirs["metrics_dir"]
	plots_dir = artifact_dirs["plots_dir"]
	examples_dir = artifact_dirs["examples_dir"]
	LOGGER.info("run_dir=%s", run_dir)
	epoch_report_rows: List[List[str]] = []
	history: List[Dict[str, Any]] = []
	loss_config = LossConfig(
		kind=config.loss_kind,
		mse_weight=config.loss_mse_weight,
		l1_weight=config.loss_l1_weight,
	)

	train_loader = create_medmnist_dataloader(
		config=MedMNISTConfig(
			dataset_name=config.dataset_name,
			split="train",
			root=config.data_root,
			download=True,
			image_size=config.image_size,
			limit_samples=config.limit_samples,
		),
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=config.num_workers,
		seed=config.seed,
	)

	eval_loader = create_medmnist_dataloader(
		config=MedMNISTConfig(
			dataset_name=config.dataset_name,
			split="val",
			root=config.data_root,
			download=True,
			image_size=config.image_size,
			limit_samples=max(1, config.limit_samples // 2),
		),
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=config.num_workers,
		seed=config.seed + 1,
	)

	first_batch = next(iter(train_loader))
	channels = _extract_images(first_batch).shape[1]

	model = TinyAutoencoder(in_channels=channels).to(device)
	optimizer = Adam(model.parameters(), lr=config.learning_rate)
	base_mask_config = SpectralMaskConfig(policy=config.mask_policy, mask_ratio=config.mask_ratio)

	scheduler: object = FixedMaskScheduler(policy=config.mask_policy, mask_ratio=config.mask_ratio)
	if config.curriculum_mode == "linear":
		scheduler = LinearMaskScheduler(
			policy=config.mask_policy,
			mask_ratio_start=config.curriculum_mask_ratio_start,
			mask_ratio_end=config.curriculum_mask_ratio_end,
			total_count=max(1, config.epochs),
			mode="epoch",
		)
	elif config.curriculum_mode in {"epoch", "step"}:
		range_size = config.curriculum_mask_ratio_end - config.curriculum_mask_ratio_start
		mid_1 = config.curriculum_mask_ratio_start + (range_size / 3.0)
		mid_2 = config.curriculum_mask_ratio_start + (2.0 * range_size / 3.0)

		if config.curriculum_mode == "step":
			total_steps = max(1, config.epochs * len(train_loader))
			stage_ends = evenly_split_stage_ends(total_count=total_steps, stage_count=3)
			stages = [
				CurriculumStage(policy="high_frequency_first", mask_ratio_start=config.curriculum_mask_ratio_start, mask_ratio_end=mid_1, end_step=stage_ends[0]),
				CurriculumStage(policy="radial", mask_ratio_start=mid_1, mask_ratio_end=mid_2, end_step=stage_ends[1]),
				CurriculumStage(policy="low_frequency_first", mask_ratio_start=mid_2, mask_ratio_end=config.curriculum_mask_ratio_end, end_step=stage_ends[2]),
			]
			scheduler = SpectralCurriculumScheduler(
				stages=stages,
				mode="step",
				default_policy="low_frequency_first",
				default_mask_ratio=config.curriculum_mask_ratio_end,
			)
		else:
			scheduler = SpectralCurriculumScheduler.high_mid_low_by_epoch(
				total_epochs=config.epochs,
				mask_ratio_start=config.curriculum_mask_ratio_start,
				mask_ratio_end=config.curriculum_mask_ratio_end,
			)

	last_train_loss = 0.0
	last_val_loss = 0.0
	best_val_loss = float("inf")
	best_epoch = -1
	stopped_early = False
	stop_epoch = max(0, config.epochs - 1)
	no_improvement_epochs = 0
	global_step = 0
	restored_best_at_end = False
	for epoch in range(config.epochs):
		if config.curriculum_mode != "step":
			epoch_mask_config = scheduler.config_at(epoch=epoch, global_step=global_step)
			last_train_loss = train_one_epoch(
				model,
				train_loader,
				optimizer,
				device,
				epoch_mask_config,
				loss_config=loss_config,
			)
			global_step += len(train_loader)
			epoch_policy = epoch_mask_config.policy
			epoch_mask_ratio = epoch_mask_config.mask_ratio
			eval_mask_config = epoch_mask_config
		else:
			model.train()
			total_loss = 0.0
			batches = 0
			last_mask_config = base_mask_config
			for batch in train_loader:
				step_mask_config = scheduler.config_at(epoch=epoch, global_step=global_step)
				images = _extract_images(batch).to(device)
				corrupted, _ = corrupt_images_in_spectral_domain(images=images, mask_config=step_mask_config)
				prediction = model(corrupted)
				loss = compute_reconstruction_loss(prediction, images, loss_config)

				optimizer.zero_grad(set_to_none=True)
				loss.backward()
				optimizer.step()

				total_loss += loss.item()
				batches += 1
				global_step += 1
				last_mask_config = step_mask_config

			last_train_loss = total_loss / max(batches, 1)
			epoch_policy = last_mask_config.policy
			epoch_mask_ratio = last_mask_config.mask_ratio
			eval_mask_config = last_mask_config

		last_val_loss = evaluate_reconstruction(
			model,
			eval_loader,
			device,
			eval_mask_config,
			loss_config=loss_config,
		)
		if last_val_loss < (best_val_loss - config.early_stopping_min_delta):
			best_val_loss = last_val_loss
			best_epoch = epoch
			no_improvement_epochs = 0
			if config.save_checkpoints:
				_save_checkpoint(
					path=checkpoints_dir / "best_val.pt",
					model=model,
					optimizer=optimizer,
					epoch=epoch,
					global_step=global_step,
					train_loss=last_train_loss,
					val_loss=last_val_loss,
					mask_policy=epoch_policy,
					mask_ratio=epoch_mask_ratio,
					config=config,
				)
		else:
			no_improvement_epochs += 1
		LOGGER.info(
			"epoch=%d policy=%s mask_ratio=%.4f train_loss=%.6f val_loss=%.6f",
			epoch,
			epoch_policy,
			epoch_mask_ratio,
			last_train_loss,
			last_val_loss,
		)

		if config.save_checkpoints and config.checkpoint_every > 0 and (epoch % config.checkpoint_every == 0):
			_save_checkpoint(
				path=checkpoints_dir / f"epoch_{epoch:03d}.pt",
				model=model,
				optimizer=optimizer,
				epoch=epoch,
				global_step=global_step,
				train_loss=last_train_loss,
				val_loss=last_val_loss,
				mask_policy=epoch_policy,
				mask_ratio=epoch_mask_ratio,
				config=config,
			)

		if config.save_examples_every > 0 and (epoch % config.save_examples_every == 0):
			_save_reconstruction_preview(
				model=model,
				dataloader=eval_loader,
				device=device,
				mask_config=eval_mask_config,
				epoch=epoch,
				output_dir=str(examples_dir),
			)

		history.append(
			{
				"epoch": epoch,
				"policy": epoch_policy,
				"mask_ratio": float(epoch_mask_ratio),
				"train_loss": float(last_train_loss),
				"val_loss": float(last_val_loss),
			}
		)

		epoch_report_rows.append(
			[
				str(epoch),
				epoch_policy,
				f"{epoch_mask_ratio:.4f}",
				f"{last_train_loss:.6f}",
				f"{last_val_loss:.6f}",
			]
		)

		if config.early_stopping_enabled and no_improvement_epochs >= config.early_stopping_patience:
			stopped_early = True
			stop_epoch = epoch
			LOGGER.info(
				"early_stopping_triggered=true epoch=%d best_epoch=%d best_val_loss=%.6f patience=%d min_delta=%.8f",
				epoch,
				best_epoch,
				best_val_loss,
				config.early_stopping_patience,
				config.early_stopping_min_delta,
			)
			break

	eval_epoch_index = stop_epoch if stopped_early else max(0, config.epochs - 1)
	if config.restore_best_at_end and config.save_checkpoints and best_epoch >= 0:
		best_checkpoint_path = checkpoints_dir / "best_val.pt"
		if best_checkpoint_path.exists():
			checkpoint = torch.load(best_checkpoint_path, map_location=device)
			model.load_state_dict(checkpoint["model_state_dict"])
			restored_best_at_end = True
			LOGGER.info(
				"restored_best_checkpoint=true path=%s best_epoch=%d best_val_loss=%.6f",
				best_checkpoint_path,
				best_epoch,
				best_val_loss,
			)
		else:
			LOGGER.warning("best checkpoint missing at end of run: %s", best_checkpoint_path)

	eval_mask_config = scheduler.config_at(epoch=eval_epoch_index, global_step=global_step)
	final_metrics = evaluate_reconstruction_metrics(model, eval_loader, device, eval_mask_config, loss_config=loss_config)
	val_loss = float(final_metrics["reconstruction_loss"])
	LOGGER.info(
		"val_reconstruction_loss=%.6f val_psnr=%.4f val_ssim=%.4f spectral_low=%.6f spectral_mid=%.6f spectral_high=%.6f",
		val_loss,
		final_metrics["psnr"],
		final_metrics["ssim"],
		final_metrics["spectral_mse_low"],
		final_metrics["spectral_mse_mid"],
		final_metrics["spectral_mse_high"],
	)

	if config.save_checkpoints:
		_save_checkpoint(
			path=checkpoints_dir / "last.pt",
			model=model,
			optimizer=optimizer,
			epoch=eval_epoch_index,
			global_step=global_step,
			train_loss=last_train_loss,
			val_loss=val_loss,
			mask_policy=eval_mask_config.policy,
			mask_ratio=eval_mask_config.mask_ratio,
			config=config,
		)

	_save_history_csv(history=history, output_path=metrics_dir / "history.csv")
	_save_loss_plot(history=history, output_path=plots_dir / "loss_curves.png")
	_save_run_report(
		run_dir=run_dir,
		history=history,
		best_epoch=best_epoch,
		best_val_loss=best_val_loss if best_epoch >= 0 else val_loss,
		final_metrics=final_metrics,
		stopped_early=stopped_early,
		stop_epoch=eval_epoch_index,
		restored_best_at_end=restored_best_at_end,
		config=config,
	)

	summary = {
		"run_dir": str(run_dir),
		"best_epoch": best_epoch,
		"best_val_loss": float(best_val_loss if best_epoch >= 0 else val_loss),
		"stopped_early": stopped_early,
		"stop_epoch": eval_epoch_index,
		"restored_best_at_end": restored_best_at_end,
		"final_val_loss": float(val_loss),
		"final_psnr": float(final_metrics["psnr"]),
		"final_ssim": float(final_metrics["ssim"]),
		"final_spectral_mse": {
			"low": float(final_metrics["spectral_mse_low"]),
			"mid": float(final_metrics["spectral_mse_mid"]),
			"high": float(final_metrics["spectral_mse_high"]),
		},
		"final_train_loss": float(last_train_loss),
		"artifact_paths": {
			"report": str(run_dir / "report.md"),
			"history_csv": str(metrics_dir / "history.csv"),
			"loss_plot": str(plots_dir / "loss_curves.png"),
			"checkpoint_last": str(checkpoints_dir / "last.pt"),
			"checkpoint_best": str(checkpoints_dir / "best_val.pt"),
			"examples_dir": str(examples_dir),
		},
	}
	(metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

	table = _format_ascii_table(
		headers=("epoch", "policy", "mask_ratio", "train_loss", "val_loss"),
		rows=epoch_report_rows,
	)
	LOGGER.info("Training report:\n%s", table)
	LOGGER.info(
		"Validation summary: split=val policy=%s mask_ratio=%.4f val_reconstruction_loss=%.6f val_psnr=%.4f val_ssim=%.4f loss_kind=%s",
		eval_mask_config.policy,
		eval_mask_config.mask_ratio,
		val_loss,
		final_metrics["psnr"],
		final_metrics["ssim"],
		config.loss_kind,
	)
	LOGGER.info("run_artifacts_saved=%s", run_dir)

	return {
		"train_loss": last_train_loss,
		"val_reconstruction_loss": val_loss,
		"val_psnr": float(final_metrics["psnr"]),
		"val_ssim": float(final_metrics["ssim"]),
		"val_spectral_mse_low": float(final_metrics["spectral_mse_low"]),
		"val_spectral_mse_mid": float(final_metrics["spectral_mse_mid"]),
		"val_spectral_mse_high": float(final_metrics["spectral_mse_high"]),
		"run_dir": str(run_dir),
	}


def parse_args() -> TrainingConfig:
	def _load_yaml_config(path: str) -> Dict[str, Any]:
		try:
			import yaml
		except ImportError as exc:  # pragma: no cover
			raise RuntimeError("PyYAML is required for --config support. Install with `pip install pyyaml`.") from exc
		with Path(path).open("r", encoding="utf-8") as handle:
			loaded = yaml.safe_load(handle) or {}
		if not isinstance(loaded, dict):
			raise ValueError(f"Expected top-level mapping in config file: {path}")
		return loaded

	def _flatten_training_yaml(raw: Dict[str, Any]) -> Dict[str, Any]:
		flat: Dict[str, Any] = {}
		dataset = raw.get("dataset", {}) if isinstance(raw.get("dataset", {}), dict) else {}
		training = raw.get("training", {}) if isinstance(raw.get("training", {}), dict) else {}
		masking = raw.get("masking", {}) if isinstance(raw.get("masking", {}), dict) else {}
		curriculum = raw.get("curriculum", {}) if isinstance(raw.get("curriculum", {}), dict) else {}
		loss = raw.get("loss", {}) if isinstance(raw.get("loss", {}), dict) else {}
		artifacts = raw.get("artifacts", {}) if isinstance(raw.get("artifacts", {}), dict) else {}
		early_stopping = raw.get("early_stopping", {}) if isinstance(raw.get("early_stopping", {}), dict) else {}
		evaluation = raw.get("evaluation", {}) if isinstance(raw.get("evaluation", {}), dict) else {}

		if "name" in dataset:
			flat["dataset_name"] = dataset["name"]
		if "root" in dataset:
			flat["data_root"] = dataset["root"]
		if "image_size" in dataset:
			flat["image_size"] = dataset["image_size"]

		if "batch_size" in training:
			flat["batch_size"] = training["batch_size"]
		if "epochs" in training:
			flat["epochs"] = training["epochs"]
		if "learning_rate" in training:
			flat["learning_rate"] = training["learning_rate"]
		if "limit_samples" in training:
			flat["limit_samples"] = training["limit_samples"]
		if "num_workers" in training:
			flat["num_workers"] = training["num_workers"]
		if "device" in training:
			flat["device"] = training["device"]
		if "seed" in training:
			flat["seed"] = training["seed"]

		if "policy" in masking:
			flat["mask_policy"] = masking["policy"]
		if "mask_ratio" in masking:
			flat["mask_ratio"] = masking["mask_ratio"]

		if "mode" in curriculum:
			flat["curriculum_mode"] = curriculum["mode"]
		if "mask_ratio_start" in curriculum:
			flat["curriculum_mask_ratio_start"] = curriculum["mask_ratio_start"]
		if "mask_ratio_end" in curriculum:
			flat["curriculum_mask_ratio_end"] = curriculum["mask_ratio_end"]

		if "kind" in loss:
			flat["loss_kind"] = loss["kind"]
		if "mse_weight" in loss:
			flat["loss_mse_weight"] = loss["mse_weight"]
		if "l1_weight" in loss:
			flat["loss_l1_weight"] = loss["l1_weight"]

		if "root" in artifacts:
			flat["artifacts_root"] = artifacts["root"]
		if "run_name" in artifacts:
			flat["run_name"] = artifacts["run_name"]
		if "save_checkpoints" in artifacts:
			flat["save_checkpoints"] = artifacts["save_checkpoints"]
		if "checkpoint_every" in artifacts:
			flat["checkpoint_every"] = artifacts["checkpoint_every"]
		if "save_examples_every" in artifacts:
			flat["save_examples_every"] = artifacts["save_examples_every"]
		if "examples_dir" in artifacts:
			flat["examples_dir_override"] = artifacts["examples_dir"]

		if "enabled" in early_stopping:
			flat["early_stopping_enabled"] = early_stopping["enabled"]
		if "patience" in early_stopping:
			flat["early_stopping_patience"] = early_stopping["patience"]
		if "min_delta" in early_stopping:
			flat["early_stopping_min_delta"] = early_stopping["min_delta"]

		if "restore_best_at_end" in evaluation:
			flat["restore_best_at_end"] = evaluation["restore_best_at_end"]

		return flat

	parser = argparse.ArgumentParser(description="Train SpecMAE prototype")
	parser.add_argument("--config", default=None)
	parser.add_argument("--dataset", default=None)
	parser.add_argument("--data-root", default=None)
	parser.add_argument("--image-size", type=int, default=None)
	parser.add_argument("--batch-size", type=int, default=None)
	parser.add_argument("--epochs", type=int, default=None)
	parser.add_argument("--lr", type=float, default=None)
	parser.add_argument("--mask-ratio", type=float, default=None)
	parser.add_argument(
		"--mask-policy",
		choices=["high_frequency_first", "low_frequency_first", "radial", "random", "high_freq", "mid_freq", "low_freq"],
		default=None,
	)
	parser.add_argument("--curriculum-mode", choices=["none", "linear", "epoch", "step"], default=None)
	parser.add_argument("--curriculum-mask-start", type=float, default=None)
	parser.add_argument("--curriculum-mask-end", type=float, default=None)
	parser.add_argument("--loss-kind", choices=["mse", "l1", "combined"], default=None)
	parser.add_argument("--loss-mse-weight", type=float, default=None)
	parser.add_argument("--loss-l1-weight", type=float, default=None)
	parser.add_argument("--artifacts-root", default=None)
	parser.add_argument("--run-name", default=None)
	parser.add_argument("--no-save-checkpoints", action="store_true")
	parser.set_defaults(no_save_checkpoints=None)
	parser.add_argument("--checkpoint-every", type=int, default=None)
	parser.add_argument("--save-examples-every", type=int, default=None)
	parser.add_argument("--examples-dir", default=None)
	parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=None)
	parser.add_argument("--early-stopping-patience", type=int, default=None)
	parser.add_argument("--early-stopping-min-delta", type=float, default=None)
	parser.add_argument("--restore-best-at-end", action=argparse.BooleanOptionalAction, default=None)
	parser.add_argument("--limit-samples", type=int, default=None)
	parser.add_argument("--num-workers", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--device", default=None)

	args = parser.parse_args()

	merged = asdict(TrainingConfig())
	if args.config:
		raw_yaml = _load_yaml_config(args.config)
		merged.update(_flatten_training_yaml(raw_yaml))

	cli_overrides = {
		"dataset_name": args.dataset,
		"data_root": args.data_root,
		"image_size": args.image_size,
		"batch_size": args.batch_size,
		"epochs": args.epochs,
		"learning_rate": args.lr,
		"mask_ratio": args.mask_ratio,
		"mask_policy": args.mask_policy,
		"curriculum_mode": args.curriculum_mode,
		"curriculum_mask_ratio_start": args.curriculum_mask_start,
		"curriculum_mask_ratio_end": args.curriculum_mask_end,
		"loss_kind": args.loss_kind,
		"loss_mse_weight": args.loss_mse_weight,
		"loss_l1_weight": args.loss_l1_weight,
		"artifacts_root": args.artifacts_root,
		"run_name": args.run_name,
		"checkpoint_every": args.checkpoint_every,
		"save_examples_every": args.save_examples_every,
		"examples_dir_override": args.examples_dir,
		"early_stopping_enabled": args.early_stopping,
		"early_stopping_patience": args.early_stopping_patience,
		"early_stopping_min_delta": args.early_stopping_min_delta,
		"restore_best_at_end": args.restore_best_at_end,
		"limit_samples": args.limit_samples,
		"num_workers": args.num_workers,
		"seed": args.seed,
		"device": args.device,
	}
	for key, value in cli_overrides.items():
		if value is not None:
			merged[key] = value

	if args.no_save_checkpoints is True:
		merged["save_checkpoints"] = False

	return TrainingConfig(**merged)


def main() -> None:
	configure_logging()
	config = parse_args()
	metrics = run_training(config)
	LOGGER.info("metrics=%s", metrics)


if __name__ == "__main__":
	main()

