"""Reconstruction evaluation entrypoint."""

from __future__ import annotations

import argparse

import torch

from specmae.data.medmnist import MedMNISTConfig, create_medmnist_dataloader
from specmae.models.mae import TinyAutoencoder
from specmae.spectral.mask import SpectralMaskConfig
from specmae.training.losses import LossConfig
from specmae.training.train import evaluate_reconstruction_metrics
from specmae.utils.logging import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate reconstruction loss for SpecMAE Phase 1")
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
	parser.add_argument("--limit-samples", type=int, default=128)
	parser.add_argument("--loss-kind", choices=["mse", "l1", "combined"], default="mse")
	parser.add_argument("--loss-mse-weight", type=float, default=1.0)
	parser.add_argument("--loss-l1-weight", type=float, default=1.0)
	parser.add_argument("--device", default="cpu")
	args = parser.parse_args()

	configure_logging()
	device = torch.device(args.device)
	loader = create_medmnist_dataloader(
		config=MedMNISTConfig(
			dataset_name=args.dataset,
			split="test",
			root=args.data_root,
			download=True,
			image_size=args.image_size,
			limit_samples=args.limit_samples,
		),
		batch_size=args.batch_size,
		shuffle=False,
	)
	first_batch = next(iter(loader))
	channels = first_batch[0].shape[1]
	model = TinyAutoencoder(in_channels=channels).to(device)
	metrics = evaluate_reconstruction_metrics(
		model=model,
		dataloader=loader,
		device=device,
		mask_config=SpectralMaskConfig(policy=args.mask_policy, mask_ratio=args.mask_ratio),
		loss_config=LossConfig(kind=args.loss_kind, mse_weight=args.loss_mse_weight, l1_weight=args.loss_l1_weight),
	)
	LOGGER.info(
		"test_metrics loss=%.6f psnr=%.4f ssim=%.4f spectral_low=%.6f spectral_mid=%.6f spectral_high=%.6f",
		metrics["reconstruction_loss"],
		metrics["psnr"],
		metrics["ssim"],
		metrics["spectral_mse_low"],
		metrics["spectral_mse_mid"],
		metrics["spectral_mse_high"],
	)


if __name__ == "__main__":
	main()

