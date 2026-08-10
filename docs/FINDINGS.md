# SpecMAE Findings (Draft for Paper)

## Scope

This document summarizes scientific findings from the completed reconstruction experiments.

Run batches analyzed:
- Curriculum hardness sweep: 20260806_163429
- Baseline vs spectral: 20260810_110815

## Primary Results

1. Curriculum masking consistently outperforms matched fixed masking across all tested spectral hardness levels on mean best validation reconstruction loss.
2. Spectral corruption substantially outperforms the current image-patch masking baseline on reconstruction loss at the tested nominal 70% masking setting.

## Spectral Curriculum Hardness Sweep

Evidence files:
- Matrix config: [configs/experiments/curriculum_hardness_sweep.yaml](../configs/experiments/curriculum_hardness_sweep.yaml)
- [outputs/experiments/curriculum_hardness_sweep_20260806_163429_leaderboard.csv](outputs/experiments/curriculum_hardness_sweep_20260806_163429_leaderboard.csv)
- [outputs/experiments/curriculum_hardness_sweep_20260806_163429_detailed.csv](outputs/experiments/curriculum_hardness_sweep_20260806_163429_detailed.csv)

Leaderboard ordering by mean_best_val_loss (lower is better):

1. linear_025_060: 0.000275214
2. linear_025_070: 0.000308850
3. linear_025_080: 0.000344879
4. linear_030_085: 0.000370298
5. fixed_060: 0.000444567
6. fixed_070: 0.000517918
7. fixed_080: 0.000755279
8. fixed_085: 0.000884101

Matched curriculum vs fixed improvements in mean_best_val_loss:

| Fixed condition | Curriculum condition | Fixed mean_best | Curriculum mean_best | Relative improvement |
|---|---|---:|---:|---:|
| fixed_060 | linear_025_060 | 0.000444567 | 0.000275214 | 38.09% |
| fixed_070 | linear_025_070 | 0.000517918 | 0.000308850 | 40.37% |
| fixed_080 | linear_025_080 | 0.000755279 | 0.000344879 | 54.34% |
| fixed_085 | linear_030_085 | 0.000884101 | 0.000370298 | 58.12% |

Interpretation:

- Curriculum gains increase as masking hardness increases in this sweep.
- At high masking levels (0.80 to 0.85 end-ratio), curriculum gives the largest relative benefit.
- Best absolute performance in this run is linear_025_060.

## Spectral vs Image-Patch Baseline

Evidence files:
- Matrix config: [configs/experiments/baseline_vs_spectral.yaml](../configs/experiments/baseline_vs_spectral.yaml)
- [outputs/experiments/baseline_vs_spectral_20260810_110815_leaderboard.csv](outputs/experiments/baseline_vs_spectral_20260810_110815_leaderboard.csv)
- [outputs/experiments/baseline_vs_spectral_20260810_110815_detailed.csv](outputs/experiments/baseline_vs_spectral_20260810_110815_detailed.csv)

Leaderboard ordering by mean_best_val_loss (lower is better):

| Condition | n | Mean best val loss | Std best val loss | Mean final val loss |
|---|---:|---:|---:|---:|
| spectral_linear_025_070 | 3 | 0.000308850 | 0.000038097 | 0.000366878 |
| spectral_fixed_070 | 3 | 0.000517918 | 0.000071906 | 0.000517918 |
| image_patch_linear_025_070 | 3 | 0.002678550 | 0.000203167 | 0.003419879 |
| image_patch_fixed_070 | 3 | 0.004501418 | 0.000316563 | 0.004565552 |

Relative improvements in mean_best_val_loss:

| Comparison | Relative improvement |
|---|---:|
| spectral_linear_025_070 vs spectral_fixed_070 | 40.37% |
| image_patch_linear_025_070 vs image_patch_fixed_070 | 40.50% |
| spectral_fixed_070 vs image_patch_fixed_070 | 88.49% |
| spectral_linear_025_070 vs image_patch_linear_025_070 | 88.47% |
| spectral_linear_025_070 vs image_patch_fixed_070 | 93.14% |

Interpretation:

- Curriculum improves both spectral and image-patch masking by roughly 40% on mean best validation reconstruction loss.
- Spectral masking remains substantially stronger than image-patch masking under both fixed and curriculum settings in this matrix.
- The current result supports a reconstruction-level advantage for spectral corruption, but does not yet establish downstream representation quality.

## Notes and Caveats

- Current evidence is from n=3 seeds per condition in the analyzed matrices.
- The current claim is limited to reconstruction validation behavior on the studied MedMNIST setting.
- Best-val loss is the primary comparison metric for this analysis.
- Same numeric mask ratios are not necessarily comparable across spectral and spatial domains; an image-patch hardness sweep is needed to choose a reconstruction-matched baseline.
- Downstream linear-probe evaluation is the next step before making representation-learning claims.
