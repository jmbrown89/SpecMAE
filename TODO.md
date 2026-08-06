# TODO

## Model Context For Spectral Corruption

- [ ] Add dilated residual blocks in the bottleneck to increase effective receptive field.
- [ ] Add a lightweight global context block at the bottleneck (for example SE/GC-style pooling).
- [ ] Evaluate a slightly deeper encoder variant and compare against current ResUNet baseline.
- [ ] Add a ViT-based reconstruction architecture variant and compare it against the ResUNet baseline.
- [ ] Add optional spectral-domain auxiliary loss while keeping image reconstruction as the primary target.

## Evaluation Additions

- [x] Add PSNR metric to training/evaluation summaries.
- [x] Add SSIM metric to training/evaluation summaries.
- [x] Add band-wise spectral reconstruction error (low/mid/high frequency bands).
- [ ] Add checkpoint-based eval CLI flow to report val/test reconstruction from saved checkpoints (for example best_val.pt), with optional checkpoint-config loading.

## Experiment Follow-Ups

- [ ] Run hardness sweep matrix and summarize mean/std by condition.
- [ ] Run cross-dataset transfer matrix (pathmnist, pneumoniamnist, octmnist).
- [ ] Run loss ablation matrix (mse vs combined) and compare effect size.
