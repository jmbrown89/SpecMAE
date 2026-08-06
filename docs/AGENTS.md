# AGENTS.md

## Project summary

This repository implements **SpecMAE**: a spectral-domain masked autoencoding framework for self-supervised medical image representation learning.

The first proof-of-concept should be small, reproducible, and easy to extend. The initial implementation should prioritise a clean experimental pipeline over architectural complexity.

The initial target is a **MedMNIST-based prototype** using a spectral masking pipeline, with the option to extend later to larger medical datasets such as BraTS.

## Core idea

SpecMAE is a masked autoencoding framework in which masking is defined in a transformed domain rather than directly in pixel space. The first implementation should use the **2D Fourier transform** as the transform basis.

High-level pipeline:

1. Load medical images.
2. Transform them into the spectral domain.
3. Apply a masking policy over spectral coefficients.
4. Invert the transform to obtain a corrupted image.
5. Train a masked autoencoder or similar reconstruction model on the corrupted input.
6. Evaluate representation quality on downstream tasks.

## Design principles

- Keep the codebase minimal and modular.
- Separate data loading, transforms, masking, model code, training, and evaluation.
- Make the transform and masking strategy pluggable so future variants can be added easily.
- Prefer small, deterministic, testable components.
- Use clear naming and avoid premature generalisation.

## Suggested repository structure

```text
specmae/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── medmnist_2d.yaml
│   ├── medmnist_3d.yaml
│   └── brats_placeholder.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   └── specmae/
│       ├── __init__.py
│       ├── data/
│       │   ├── medmnist.py
│       │   └── transforms.py
│       ├── spectral/
│       │   ├── fft.py
│       │   ├── mask.py
│       │   └── inverse.py
│       ├── models/
│       │   ├── mae.py
│       │   ├── encoder.py
│       │   └── decoder.py
│       ├── training/
│       │   ├── train.py
│       │   ├── losses.py
│       │   └── scheduler.py
│       ├── evaluation/
│       │   ├── downstream.py
│       │   └── reconstruction.py
│       └── utils/
│           ├── seed.py
│           ├── logging.py
│           └── metrics.py
├── scripts/
│   ├── train_medmnist.sh
│   ├── eval_medmnist.sh
│   └── visualize_masks.py
├── tests/
│   ├── test_fft_roundtrip.py
│   ├── test_masking.py
│   ├── test_dataset_loading.py
│   └── test_training_step.py
└── notebooks/
    └── exploration.ipynb
```

If the repository already has a preferred structure, adapt to it rather than forcing this exact layout.

## Implementation priorities

### Phase 1: working proof of concept

Build the smallest end-to-end version that can:

- load a MedMNIST dataset,
- apply Fourier transform masking,
- reconstruct the image,
- run one training loop,
- produce a basic evaluation metric.

The first goal is not state of the art. The first goal is to verify that the spectral masking idea behaves sensibly.

### Phase 2: curriculum masking

Add the progressive masking schedule that distinguishes SpecMAE from a simple frequency-masked autoencoder.

The schedule should support at least:

- masking high-frequency coefficients first,
- then expanding to mid frequencies,
- then including low frequencies.

### Phase 3: comparative experiments

Add baselines and ablations:

- vanilla MAE or image-space masking baseline,
- random spectral masking,
- fixed band masking,
- curriculum masking,
- alternate transform backends if the codebase supports them.

## Coding requirements

- Use type hints where practical.
- Prefer pure functions for transform and masking utilities.
- Keep dataset-specific code isolated from model code.
- Write small, focused functions.
- Avoid hidden side effects.
- Add docstrings for non-obvious behaviour.
- Ensure all random number generation is seedable.

## Spectral transform requirements

For the first implementation, the Fourier path should:

- support real-valued image inputs,
- handle complex-valued coefficients explicitly,
- preserve conjugate symmetry where needed,
- allow coefficient masking by band or radial shell,
- support inverse reconstruction back to image space.

The transform module should not be hard-wired to one masking scheme.

## Curriculum requirements

The scheduler should be configurable.

Minimum supported options:

- fixed stages by epoch,
- fixed stages by step count,
- progressive band activation,
- optional gradual increase in mask ratio.

The curriculum logic should be isolated from the transform logic.

## Evaluation requirements

At minimum, report:

- reconstruction loss,
- downstream classification accuracy or AUC if applicable,
- qualitative reconstruction examples,
- ablations for masking strategy and curriculum order.

If the dataset and task allow, include:

- robustness to corruption,
- calibration or uncertainty metrics,
- transfer performance across splits.

## Testing requirements

Add tests that verify:

- FFT forward/inverse round-trip behaves as expected,
- masking preserves tensor shape and dtype expectations,
- curriculum scheduling activates bands in the correct order,
- a single training step runs without errors,
- dataset loading returns the expected sample structure.

## Data handling

- Do not commit raw datasets.
- Document how to obtain MedMNIST data.
- Cache processed data locally if needed.
- Keep any BraTS-specific code separate until the prototype is stable.

## Documentation requirements

The README should include:

- a short description of SpecMAE,
- the motivation for spectral masking,
- setup instructions,
- how to run training,
- how to run evaluation,
- one example command for the MedMNIST proof of concept,
- notes on how the method differs from PMAE.

## Suggested first implementation milestones

1. Create the project skeleton and packaging.
2. Implement Fourier transform utilities.
3. Implement spectral masking and inverse reconstruction.
4. Build a tiny training loop on MedMNIST.
5. Add a curriculum scheduler.
6. Add a simple evaluation script.
7. Add tests for the critical utilities.
8. Add plots or visualisations of masking and reconstructions.

## Notes for future expansion

The code should be organised so that future work can swap in other transforms, such as:

- wavelets,
- PCA-like transforms,
- diffusion-style embeddings.

Do not design for all of those now. Just keep the interface open enough that they could be added later without rewriting the whole project.
