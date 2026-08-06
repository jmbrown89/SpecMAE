# SpecMAE Usage Guide

This guide is a practical reference for running training, experiments, and evaluation in this repository.

## 1. Environment Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Notes:
- The first data-using run downloads MedMNIST into `data/raw`.
- Most default configs are CPU-safe for smoke tests, but longer runs are slower on CPU.

## 2. Core Commands

The repository exposes a top-level CLI:

```powershell
specmae --help
```

Main subcommands:
- `specmae train`
- `specmae eval`
- `specmae matrix`
- `specmae downstream`

## 3. Train A Reconstruction Model

### 3.1 Spectral method (default)

```powershell
specmae train -c configs/medmnist_2d.yaml
```

### 3.2 Baseline MAE-style image masking

```powershell
specmae train -c configs/medmnist_2d_baseline_mae.yaml
```

### 3.3 Override config values from CLI

```powershell
specmae train -c configs/medmnist_2d.yaml --epochs 10 --run-name my_run --limit-samples 512
```

Useful training options:
- `--pretext-method spectral|image_patch`
- `--mask-ratio <float>`
- `--mask-policy high_frequency_first|low_frequency_first|radial|random`
- `--image-patch-size <int>` (used for `image_patch` method)
- `--curriculum-mode none|linear|epoch|step`
- `--save-examples-every <int>`
- `--early-stopping`
- `--restore-best-at-end`

## 4. Run Experiment Matrices

### 4.1 Full curriculum matrix

```powershell
specmae matrix -m configs/experiments/curriculum_matrix.yaml
```

### 4.2 Hardness sweep

```powershell
specmae matrix -m configs/experiments/curriculum_hardness_sweep.yaml
```

### 4.3 Resume interrupted matrix

```powershell
specmae matrix -m configs/experiments/curriculum_matrix.yaml --resume
```

Resume behavior:
- Reuses completed runs from the selected batch timestamp.
- If a partial run folder exists without `metrics/summary.json`, it is reset and rerun.

## 5. Evaluate Reconstruction Metrics

The eval command reports:
- Reconstruction loss
- PSNR
- SSIM
- Spectral MSE (low/mid/high bands)

### 5.1 Evaluate with explicit settings

```powershell
specmae eval --dataset pathmnist --split test --mask-policy high_frequency_first --mask-ratio 0.6 --limit-samples 256
```

### 5.2 Evaluate from checkpoint (recommended)

```powershell
specmae eval --checkpoint outputs/runs/<run_name>/checkpoints/best_val.pt --use-checkpoint-config --split test
```

This loads checkpoint weights and can inherit dataset/masking/loss settings from the checkpoint config.

## 6. Run Downstream Linear Probe

This evaluates representation quality by freezing the encoder and training a linear classifier.

### 6.1 Basic downstream probe

```powershell
specmae downstream --dataset pathmnist --limit-train 2048 --limit-test 1024 --probe-epochs 50
```

### 6.2 Probe from a trained checkpoint

```powershell
specmae downstream --checkpoint outputs/runs/<run_name>/checkpoints/best_val.pt --use-checkpoint-config --limit-train 2048 --limit-test 1024
```

Outputs include train/test accuracy and binary AUC when applicable.

## 7. Output Structure

Training artifacts are written under:

```text
outputs/runs/<run_name>/
```

Typical contents:
- `checkpoints/best_val.pt`
- `checkpoints/last.pt`
- `metrics/history.csv`
- `metrics/summary.json`
- `plots/loss_curves.png`
- `examples/epoch_*.png`
- `report.md`

Matrix summaries are written under:

```text
outputs/experiments/
```

Typical contents:
- `*_detailed.csv`
- `*_leaderboard.csv`
- `*_summary.md`

## 8. Suggested Repro Workflow

1. Train spectral candidate(s):
   - `specmae train -c configs/medmnist_2d.yaml`
2. Train baseline MAE candidate(s):
   - `specmae train -c configs/medmnist_2d_baseline_mae.yaml`
3. Evaluate reconstruction from best checkpoints:
   - `specmae eval --checkpoint ... --use-checkpoint-config --split test`
4. Evaluate downstream quality from same checkpoints:
   - `specmae downstream --checkpoint ... --use-checkpoint-config`
5. Record results in `FINDINGS.md`.

## 9. Troubleshooting

- If a matrix run stops mid-way, rerun with `--resume`.
- If no examples appear, check `save_examples_every` in the matrix/training config.
- If YAML parsing fails, ensure space indentation (no tabs).
- If command not found for `specmae`, run `pip install -e .` in the activated environment.
