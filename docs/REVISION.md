# SpecMAE prototype revision

## Goal

The current prototype demonstrates that aggressive spectral masking is substantially harder than conventional MAE-style masking. The objective is **not** to make the task easier by weakening the corruption, but to give the model an architecture better suited to reconstructing globally corrupted inputs.

The focus remains on validating the SpecMAE concept on MedMNIST before scaling to larger datasets.

---

## 1. Replace the backbone

Remove the current simple baseline encoder/decoder.

Replace it with a lightweight **ResUNet** architecture.

Requirements:

* ResNet-18 encoder (or equivalent lightweight residual encoder)
* Small U-Net style decoder
* Skip connections retained
* Final reconstruction head outputs a single reconstructed image

The architecture should remain relatively small since MedMNIST images are only 28×28.

The encoder should ultimately be reusable for downstream classification.

---

## 2. Preserve the SpecMAE pipeline

The overall pipeline should remain:

```
Input image
      │
      ▼
2D FFT
      │
      ▼
Spectral masking
      │
      ▼
Inverse FFT
      │
      ▼
Corrupted image
      │
      ▼
ResUNet
      │
      ▼
Reconstructed image
```

Do **not** move masking into image space.

Do **not** perform MAE-style patch masking.

The only corruption mechanism should be spectral masking.

---

## 3. Improve Fourier masking implementation

Review the masking implementation.

Ensure that:

* complex-valued FFT coefficients are handled correctly
* Hermitian symmetry is preserved
* inverse FFT always produces a valid real-valuated image
* masking is applied consistently to conjugate coefficient pairs

If these are already implemented, verify correctness.

---

## 4. Support configurable masking strategies

Refactor the masking module so multiple strategies can easily be compared.

Implement:

* random coefficient masking
* radial masking
* high-frequency-first masking
* low-frequency-first masking

These should be selectable via configuration.

---

## 5. Add curriculum scheduler

Implement a masking scheduler independent of the model.

Initially support:

* fixed mask ratio
* linear increase in mask ratio
* staged curriculum

The scheduler should control:

* active frequency bands
* mask ratio

without modifying the model.

---

## 6. Improve losses

Current reconstruction loss should remain.

Add optional support for:

* image-space MSE
* image-space L1
* combined loss

Structure the code so frequency-domain losses can easily be added later.

---

## 7. Visualisation

Add utilities to visualise:

* original image
* FFT magnitude
* masked spectrum
* reconstructed spectrum
* corrupted image after inverse FFT
* reconstructed output

This is likely to be invaluable when debugging masking strategies.

---

## 8. Logging

Record:

* reconstruction loss
* validation loss
* current mask ratio
* current curriculum stage
* representative reconstruction images every N epochs

---

## 9. Keep the code transform-agnostic

Although only FFT is required initially, isolate the transform interface.

Something like:

```
Transform
    forward()
    inverse()
```

should make it straightforward to later add:

* Wavelet
* PCA
* Diffusion Maps

without changing the remainder of the pipeline.

---

## 10. Preserve simplicity

Do **not** introduce:

* attention mechanisms
* GAN losses
* diffusion models
* adversarial training
* complicated auxiliary networks

The goal is to isolate whether **spectral-domain masking alone** improves representation learning.

---

## 11. Future-proof the implementation

Organise the code so that SpecMAE is treated as a **framework**, with FFT as its first implementation.

For example:

```
specmae/
    transforms/
        fft.py
        base.py
    masking/
        random.py
        radial.py
        curriculum.py
```

This will make it straightforward to compare alternative transforms later, as suggested by the PMAE authors.

---

One additional change I'd recommend, based on our discussion, is to make the transform and masking components **strictly decoupled**. The transform should only know how to convert between image space and coefficient space. The masking policy should operate purely on coefficients and know nothing about FFT internals. This separation will make it much easier to answer the scientific question your paper is really asking: *does the choice of transformation matter, or is it the masking policy itself?*
