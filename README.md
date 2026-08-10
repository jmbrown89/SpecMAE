# SpecMAE

SpecMAE is a spectral-domain masked autoencoding prototype for self-supervised medical image learning.
This Phase 1 implementation focuses on a small, reproducible MedMNIST proof of concept.

## Why Spectral Masking

Instead of masking image-space patches directly, SpecMAE masks Fourier coefficients and reconstructs from a spectrally corrupted input.
This allows experiments that target frequency content explicitly.

The implementation now separates transform and masking concerns:
- `specmae.transforms`: image <-> coefficient backends (FFT first)
- `specmae.masking`: masking policies and curriculum schedulers

## Phase 1 Scope

- MedMNIST loading
- 2D FFT masking and inverse reconstruction
- Lightweight ResUNet reconstruction model and training loop
- Basic reconstruction-loss evaluation

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train (MedMNIST Proof of Concept)

```bash
python -m specmae.training.train --dataset pathmnist --epochs 1 --limit-samples 256 --mask-policy high_frequency_first --mask-ratio 0.6
```

Top-level CLI equivalent:

```bash
specmae train -c configs/medmnist_2d.yaml
```

Config-driven training:

```bash
python -m specmae.training.train --config configs/medmnist_2d.yaml
```

CLI flags override YAML values when both are provided:

```bash
python -m specmae.training.train --config configs/medmnist_2d.yaml --epochs 10 --run-name medmnist_override
```

The first run downloads MedMNIST files to `data/raw`.

## Evaluate Reconstruction

```bash
python -m specmae.evaluation.reconstruction --dataset pathmnist --limit-samples 128 --mask-policy high_frequency_first --mask-ratio 0.6
```

Top-level CLI equivalent:

```bash
specmae eval --dataset pathmnist --limit-samples 128 --mask-policy high_frequency_first --mask-ratio 0.6
```

## Masking And Curriculum Options

- Mask policies: `high_frequency_first`, `low_frequency_first`, `radial`, `random`
- Scheduler modes: `none` (fixed), `linear`, `epoch` (staged), `step` (staged)
- Loss modes: `mse`, `l1`, `combined`

Example:

```bash
python -m specmae.training.train --dataset pathmnist --epochs 5 --curriculum-mode linear --curriculum-mask-start 0.2 --curriculum-mask-end 0.8 --loss-kind combined --loss-mse-weight 1.0 --loss-l1-weight 0.25 --save-examples-every 1
```

## Reviewing Completed Runs

Training now saves a run artifact bundle by default under `outputs/runs/<run_name_or_timestamp>/`.

Example:

```bash
python -m specmae.training.train --dataset pneumoniamnist --epochs 8 --run-name pneumo_resunet_v1
```

Saved outputs include:

- `checkpoints/last.pt`
- `checkpoints/best_val.pt`
- `checkpoints/epoch_*.pt`
- `metrics/history.csv`
- `metrics/summary.json`
- `plots/loss_curves.png`
- `examples/epoch_*.png`
- `report.md`

Useful options:

- `--artifacts-root outputs/runs`
- `--run-name my_experiment`
- `--checkpoint-every 1`
- `--save-examples-every 1`
- `--examples-dir <path>`
- `--no-save-checkpoints`

## Experiment Matrix Runner

To run seeded ablations and generate a combined leaderboard:

```bash
python scripts/run_experiment_matrix.py --matrix configs/experiments/curriculum_matrix.yaml
```

Top-level CLI equivalent:

```bash
specmae matrix -m configs/experiments/curriculum_matrix.yaml
```

For a quick smoke version:

```bash
python scripts/run_experiment_matrix.py --matrix configs/experiments/curriculum_matrix_smoke.yaml
```

Smoke matrices clean their generated run artifacts and result files after a successful run.

To sweep the image-patch baseline across several spatial masking hardness levels:

```bash
python scripts/run_experiment_matrix.py --matrix configs/experiments/image_patch_hardness_sweep.yaml
```

Outputs are written to `outputs/experiments/` as:

- `*_detailed.csv` (per run)
- `*_leaderboard.csv` (aggregated by condition)
- `*_summary.md` (human-readable ranking)

## How This Differs From PMAE

This prototype prioritizes spectral-domain corruption using Fourier masks.
PMAE-style approaches are primarily patch-space masking methods, while SpecMAE directly manipulates transform-domain coefficients.

