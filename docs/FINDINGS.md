# SpecMAE Findings (Draft for Paper)

## Scope

This document summarizes scientific findings from the latest full curriculum hardness sweep.

Run batch analyzed: 20260806_163429

## Primary Result

Curriculum masking consistently outperforms matched fixed masking across all tested hardness levels on mean best validation reconstruction loss.

Evidence files:
- [outputs/experiments/curriculum_hardness_sweep_20260806_163429_leaderboard.csv](outputs/experiments/curriculum_hardness_sweep_20260806_163429_leaderboard.csv)
- [outputs/experiments/curriculum_hardness_sweep_20260806_163429_detailed.csv](outputs/experiments/curriculum_hardness_sweep_20260806_163429_detailed.csv)

## Quantitative Findings

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

## Interpretation

- Curriculum gains increase as masking hardness increases in this sweep.
- At high masking levels (0.80 to 0.85 end-ratio), curriculum gives the largest relative benefit.
- Best absolute performance in this run is linear_025_060.

## Notes and Caveats

- Current evidence is from n=3 seeds per condition in the hardness sweep.
- The current claim is limited to reconstruction validation behavior on the studied MedMNIST setting.
- Best-val loss is the primary comparison metric for this analysis.
