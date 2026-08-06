#!/usr/bin/env bash
set -euo pipefail

python -m specmae.training.train \
	--config configs/medmnist_2d.yaml

