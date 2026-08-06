#!/usr/bin/env bash
set -euo pipefail

python -m specmae.evaluation.reconstruction \
	--dataset pathmnist \
	--limit-samples 128 \
	--mask-policy high_freq \
	--mask-ratio 0.6

