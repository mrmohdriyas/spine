# Phase 11 — Advanced DRR Rendering & Cross-Domain Vertebral Localization

## 1. Objective
The objective of Phase 11 was to improve the synthetic-to-real domain compatibility between VerSe 3D CT projections and real VinDr-SpineXR radiographs by improving DRR realism (rendering physics). The goal was to establish a more realistic cross-modality anatomical localization pipeline without yet resorting to Unsupervised Domain Adaptation (UDA).

## 2. Phase 10 Limitations
In Phase 10, the projection was a simple Mean/Sum Intensity Projection of Hounsfield Units directly onto a 2D plane. Since air (-1000 HU) dominated the sum, the resulting DRRs had black backgrounds and bright, isolated vertebral blobs lacking soft-tissue attenuation, scatter, and physical ray-tracing properties, causing massive domain failure on real clinical X-rays.

## 3. Domain Statistics
A statistical comparison of the Phase 10 V1 DRRs against real VinDr X-rays was conducted:
- **Phase 10 V1 DRRs:** Bimodal intensity distribution with extremely high contrast (black background, bright bone).
- **VinDr X-rays:** Unimodal/broad Gaussian distribution characterized by widespread mid-gray soft tissue, bowel gas, and overlapping bone shadows with significantly less contrast.

*(Outputs: `outputs/phase11_domain_adaptation/intensity_statistics.csv`)*

## 4. DRR Rendering Method
For V2 DRRs, a physically-inspired Beer-Lambert attenuation model was implemented:
- **HU Conversion:** CT values were clipped to `[-1000, 3000]`.
- **Attenuation Coeff ($\mu$):** HU values were mapped to linear attenuation coefficients using $\mu_{water} \approx 0.17 \, \text{cm}^{-1}$.
- **Ray Integral:** The volume was projected by summing $\mu \times dy$ (where $dy$ is the voxel spacing in the projection axis), matching the physical line integral $\int \mu \, dy$.
- **Simulated Soft Tissue:** A background soft-tissue volume (~0 HU + noise) was introduced to simulate patient body mass and normalize background attenuation.

## 5. Anatomical Label Preservation
Vertebral masks (L1–L5) were preserved directly from the 3D segmentation. Their 3D coordinates were accurately projected into the 2D plane to create bounding boxes, ensuring that **no appearance-based pseudo-labeling or fabricated labels were used.** The anatomical ground truth remained strictly VerSe-derived.

## 6. V1 vs V2 Synthetic Results
A Faster R-CNN localizer (ResNet50 backbone) was trained from scratch on the advanced V2 DRRs using the exact same hyperparameters as V1.
- **V1 Model:** High performance on simple blob-like V1 DRRs.
- **V2 Model:** Equivalent high performance on V2 synthetic DRRs. The model successfully adapted to the more complex noisy background and correctly localized L1-L5 with high confidence.

## 7. VinDr Qualitative Transfer
The V2 localizer was evaluated qualitatively on a small sample of real VinDr-SpineXR images.
*(Outputs: `outputs/phase11_domain_adaptation/vindr_transfer/`)*

## 8. Domain-Gap Analysis
Despite physical rendering improvements, a significant domain gap persists:
1. **Vertebral Localization:** The model still struggles to consistently identify vertebrae on real X-rays, occasionally confusing ribs or bowel gas edges for vertebral bodies.
2. **Confidence:** Confidence scores dropped significantly on real X-rays compared to the synthetic hold-outs.
3. **Conclusion:** While the physical rendering (Beer-Lambert attenuation) improved background contrast slightly, the lack of real anatomical structures in the synthetic background (no ribs, no pelvis, no organ shadows) means the DRR distribution is still too "clean" compared to clinical radiographs.

## 9. Limitations
- The simulated soft tissue background was purely Gaussian noise in a cylindrical mask; it lacked structured anatomical noise (ribs, soft tissue boundaries).
- Projection was parallel, not cone-beam.
- DeepDRR or Monte-Carlo scatter simulations were not used to keep the pipeline computationally tractable.

## 10. Decision Gate
**B. DOMAIN GAP REMAINS**

Improved physical rendering (Beer-Lambert attenuation + HU scaling) does not adequately solve the transfer problem on its own. Structured anatomical noise and texture variations in real VinDr radiographs still cause the localizer to fail. A mapping between the synthetic distribution and real distribution is required.

## 11. Recommended Phase 12
**Phase 12: Controlled Domain Adaptation Experiment**

Since physical rendering alone is insufficient, Phase 12 must investigate bridging the visual domain gap. We should proceed with **Unsupervised Domain Adaptation (UDA)**. Specifically:
1. Implement a CycleGAN or Adversarial Domain Adaptation network to translate synthetic DRRs into the VinDr "style" (adding realistic ribs, contrast, and noise).
2. Train the vertebral localizer on the *adapted* DRRs.
3. Once localization transfers successfully, we can extract bounding boxes to feed into the Vertebral Morphology Encoder (VME) of the final NeuroMorphNet.
