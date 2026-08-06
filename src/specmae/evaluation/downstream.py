"""Downstream linear-probe evaluation for learned representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from specmae.data.medmnist import MedMNISTConfig, create_medmnist_dataloader
from specmae.models.mae import TinyAutoencoder
from specmae.utils.logging import configure_logging, get_logger


LOGGER = get_logger(__name__)


def _resolve_class_count(dataset: str, labels: torch.Tensor) -> int:
	"""Resolve class count from MedMNIST metadata, with tensor fallback."""
	try:
		from medmnist import INFO  # type: ignore
		raw = INFO.get(dataset, {}).get("label", {})
		if isinstance(raw, dict) and len(raw) > 0:
			return int(len(raw))
	except Exception:
		pass
	return int(labels.max().item()) + 1


def _apply_checkpoint_config_overrides(
	dataset: str,
	data_root: str,
	image_size: int,
	checkpoint_payload: Dict[str, Any],
	enabled: bool,
) -> Tuple[str, str, int]:
	if not enabled:
		return dataset, data_root, image_size
	raw_config = checkpoint_payload.get("config", {})
	if not isinstance(raw_config, dict):
		return dataset, data_root, image_size
	return (
		str(raw_config.get("dataset_name", dataset)),
		str(raw_config.get("data_root", data_root)),
		int(raw_config.get("image_size", image_size)),
	)


def _extract_images_labels(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
	images, labels = batch
	if labels.ndim > 1:
		labels = labels[:, 0]
	return images, labels.long()


@torch.no_grad()
def _extract_features(
	encoder: nn.Module,
	loader: torch.utils.data.DataLoader,
	device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
	encoder.eval()
	features = []
	labels_all = []
	for batch in loader:
		images, labels = _extract_images_labels(batch)
		images = images.to(device)
		labels = labels.to(device)
		feat_map = encoder(images)
		feat_vec = F.adaptive_avg_pool2d(feat_map, output_size=1).flatten(1)
		features.append(feat_vec)
		labels_all.append(labels)
	return torch.cat(features, dim=0), torch.cat(labels_all, dim=0)


def _binary_auc(prob_positive: torch.Tensor, labels: torch.Tensor) -> float:
	"""Compute binary AUC using rank statistic."""
	labels = labels.long()
	pos_mask = labels == 1
	neg_mask = labels == 0
	n_pos = int(pos_mask.sum().item())
	n_neg = int(neg_mask.sum().item())
	if n_pos == 0 or n_neg == 0:
		return float("nan")

	order = torch.argsort(prob_positive)
	ranks = torch.empty_like(order, dtype=torch.float32)
	ranks[order] = torch.arange(1, prob_positive.numel() + 1, device=prob_positive.device, dtype=torch.float32)
	sum_ranks_pos = ranks[pos_mask].sum()
	auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
	return float(auc.item())


def run_linear_probe(
	dataset: str,
	data_root: str,
	image_size: int,
	batch_size: int,
	limit_train: int,
	limit_test: int,
	epochs: int,
	learning_rate: float,
	device: torch.device,
	checkpoint: str,
	use_checkpoint_config: bool,
) -> Dict[str, float]:
	checkpoint_payload: Dict[str, Any] = {}
	if checkpoint:
		checkpoint_payload = torch.load(checkpoint, map_location=device)
		dataset, data_root, image_size = _apply_checkpoint_config_overrides(
			dataset=dataset,
			data_root=data_root,
			image_size=image_size,
			checkpoint_payload=checkpoint_payload,
			enabled=use_checkpoint_config,
		)

	train_loader = create_medmnist_dataloader(
		config=MedMNISTConfig(
			dataset_name=dataset,
			split="train",
			root=data_root,
			download=True,
			image_size=image_size,
			limit_samples=limit_train,
		),
		batch_size=batch_size,
		shuffle=True,
	)
	test_loader = create_medmnist_dataloader(
		config=MedMNISTConfig(
			dataset_name=dataset,
			split="test",
			root=data_root,
			download=True,
			image_size=image_size,
			limit_samples=limit_test,
		),
		batch_size=batch_size,
		shuffle=False,
	)

	first_batch = next(iter(train_loader))
	channels = first_batch[0].shape[1]
	autoencoder = TinyAutoencoder(in_channels=channels).to(device)
	if checkpoint_payload:
		autoencoder.load_state_dict(checkpoint_payload["model_state_dict"], strict=False)
		LOGGER.info("loaded_checkpoint=%s", checkpoint)

	encoder = autoencoder.encoder
	for param in encoder.parameters():
		param.requires_grad = False

	train_features, train_labels = _extract_features(encoder=encoder, loader=train_loader, device=device)
	test_features, test_labels = _extract_features(encoder=encoder, loader=test_loader, device=device)

	n_classes = _resolve_class_count(dataset=dataset, labels=train_labels)
	probe = nn.Linear(train_features.shape[1], n_classes).to(device)
	optimizer = Adam(probe.parameters(), lr=learning_rate)

	for _ in range(epochs):
		probe.train()
		logits = probe(train_features)
		loss = F.cross_entropy(logits, train_labels)
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

	probe.eval()
	with torch.no_grad():
		train_logits = probe(train_features)
		test_logits = probe(test_features)
		train_pred = train_logits.argmax(dim=1)
		test_pred = test_logits.argmax(dim=1)
		train_acc = float((train_pred == train_labels).float().mean().item())
		test_acc = float((test_pred == test_labels).float().mean().item())
		test_auc = float("nan")
		if n_classes == 2:
			test_prob = torch.softmax(test_logits, dim=1)[:, 1]
			test_auc = _binary_auc(test_prob, test_labels)

	return {
		"train_accuracy": train_acc,
		"test_accuracy": test_acc,
		"test_auc": test_auc,
		"num_classes": float(n_classes),
		"num_train_samples": float(train_labels.numel()),
		"num_test_samples": float(test_labels.numel()),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Run downstream linear-probe evaluation")
	parser.add_argument("--dataset", default="pathmnist")
	parser.add_argument("--data-root", default="data/raw")
	parser.add_argument("--image-size", type=int, default=28)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--limit-train", type=int, default=2048)
	parser.add_argument("--limit-test", type=int, default=1024)
	parser.add_argument("--probe-epochs", type=int, default=50)
	parser.add_argument("--probe-lr", type=float, default=1e-2)
	parser.add_argument("--checkpoint", default="")
	parser.add_argument("--use-checkpoint-config", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--output-json", default="")
	parser.add_argument("--device", default="cpu")
	args = parser.parse_args()

	configure_logging()
	device = torch.device(args.device)
	metrics = run_linear_probe(
		dataset=args.dataset,
		data_root=args.data_root,
		image_size=args.image_size,
		batch_size=args.batch_size,
		limit_train=args.limit_train,
		limit_test=args.limit_test,
		epochs=args.probe_epochs,
		learning_rate=args.probe_lr,
		device=device,
		checkpoint=args.checkpoint,
		use_checkpoint_config=bool(args.use_checkpoint_config),
	)
	LOGGER.info("downstream_metrics=%s", metrics)

	if args.output_json.strip():
		out_path = Path(args.output_json)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
		LOGGER.info("saved_downstream_metrics=%s", out_path)


if __name__ == "__main__":
	main()

