"""Reconstruction evaluation entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict

import torch

from specmae.data.medmnist import MedMNISTConfig, create_medmnist_dataloader
from specmae.models.mae import TinyAutoencoder
from specmae.spectral.mask import SpectralMaskConfig
from specmae.training.losses import LossConfig
from specmae.training.train import evaluate_reconstruction_metrics
from specmae.utils.logging import configure_logging, get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class EvalConfig:
	dataset: str
	data_root: str
	image_size: int
	batch_size: int
	mask_ratio: float
	mask_policy: str
	pretext_method: str
	image_patch_size: int
	limit_samples: int
	loss_kind: str
	loss_mse_weight: float
	loss_l1_weight: float
	device: str
	split: str
	checkpoint: str
	use_checkpoint_config: bool


def _merge_checkpoint_overrides(base: EvalConfig, checkpoint_payload: Dict[str, Any]) -> EvalConfig:
	if not base.use_checkpoint_config:
		return base

	raw_config = checkpoint_payload.get("config", {})
	if not isinstance(raw_config, dict):
		LOGGER.warning("Checkpoint config missing or invalid; skipping config override")
		return base

	overrides = {
		"dataset": raw_config.get("dataset_name", base.dataset),
		"data_root": raw_config.get("data_root", base.data_root),
		"image_size": int(raw_config.get("image_size", base.image_size)),
		"mask_ratio": float(raw_config.get("mask_ratio", base.mask_ratio)),
		"mask_policy": str(raw_config.get("mask_policy", base.mask_policy)),
		"pretext_method": str(raw_config.get("pretext_method", base.pretext_method)),
		"image_patch_size": int(raw_config.get("image_patch_size", base.image_patch_size)),
		"loss_kind": str(raw_config.get("loss_kind", base.loss_kind)),
		"loss_mse_weight": float(raw_config.get("loss_mse_weight", base.loss_mse_weight)),
		"loss_l1_weight": float(raw_config.get("loss_l1_weight", base.loss_l1_weight)),
	}
	return EvalConfig(**{**base.__dict__, **overrides})


def _parse_args() -> EvalConfig:
	parser = argparse.ArgumentParser(description="Evaluate reconstruction metrics for SpecMAE")
	parser.add_argument("--dataset", default="pathmnist")
	parser.add_argument("--data-root", default="data/raw")
	parser.add_argument("--image-size", type=int, default=28)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--mask-ratio", type=float, default=0.6)
	parser.add_argument(
		"--mask-policy",
		choices=["high_frequency_first", "low_frequency_first", "radial", "random", "high_freq", "mid_freq", "low_freq"],
		default="high_frequency_first",
	)
	parser.add_argument("--pretext-method", choices=["spectral", "image_patch"], default="spectral")
	parser.add_argument("--image-patch-size", type=int, default=4)
	parser.add_argument("--limit-samples", type=int, default=128)
	parser.add_argument("--loss-kind", choices=["mse", "l1", "combined"], default="mse")
	parser.add_argument("--loss-mse-weight", type=float, default=1.0)
	parser.add_argument("--loss-l1-weight", type=float, default=1.0)
	parser.add_argument("--split", choices=["train", "val", "test"], default="test")
	parser.add_argument("--checkpoint", default="")
	parser.add_argument("--use-checkpoint-config", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--device", default="cpu")
	args = parser.parse_args()

	return EvalConfig(
		dataset=args.dataset,
		data_root=args.data_root,
		image_size=args.image_size,
		batch_size=args.batch_size,
		mask_ratio=args.mask_ratio,
		mask_policy=args.mask_policy,
		pretext_method=args.pretext_method,
		image_patch_size=args.image_patch_size,
		limit_samples=args.limit_samples,
		loss_kind=args.loss_kind,
		loss_mse_weight=args.loss_mse_weight,
		loss_l1_weight=args.loss_l1_weight,
		device=args.device,
		split=args.split,
		checkpoint=args.checkpoint,
		use_checkpoint_config=bool(args.use_checkpoint_config),
	)


def main() -> None:
	configure_logging()
	config = _parse_args()
	device = torch.device(config.device)
	checkpoint_payload: Dict[str, Any] = {}
	if config.checkpoint.strip():
		checkpoint_payload = torch.load(config.checkpoint, map_location=device)
		config = _merge_checkpoint_overrides(config, checkpoint_payload)

	loader = create_medmnist_dataloader(
		config=MedMNISTConfig(
			dataset_name=config.dataset,
			split=config.split,
			root=config.data_root,
			download=True,
			image_size=config.image_size,
			limit_samples=config.limit_samples,
		),
		batch_size=config.batch_size,
		shuffle=False,
	)
	first_batch = next(iter(loader))
	channels = first_batch[0].shape[1]
	model = TinyAutoencoder(in_channels=channels).to(device)
	if checkpoint_payload:
		model.load_state_dict(checkpoint_payload["model_state_dict"], strict=False)
		LOGGER.info("loaded_checkpoint=%s", config.checkpoint)

	metrics = evaluate_reconstruction_metrics(
		model=model,
		dataloader=loader,
		device=device,
		mask_config=SpectralMaskConfig(policy=config.mask_policy, mask_ratio=config.mask_ratio),
		pretext_method=config.pretext_method,
		image_patch_size=config.image_patch_size,
		loss_config=LossConfig(kind=config.loss_kind, mse_weight=config.loss_mse_weight, l1_weight=config.loss_l1_weight),
	)
	LOGGER.info(
		"reconstruction_metrics split=%s loss=%.6f psnr=%.4f ssim=%.4f spectral_low=%.6f spectral_mid=%.6f spectral_high=%.6f",
		config.split,
		metrics["reconstruction_loss"],
		metrics["psnr"],
		metrics["ssim"],
		metrics["spectral_mse_low"],
		metrics["spectral_mse_mid"],
		metrics["spectral_mse_high"],
	)


if __name__ == "__main__":
	main()

