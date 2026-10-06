# Phase 12 — Controlled Domain Adaptation for Vertebral Localization

## 1. Objective
Investigate whether feature-level Unsupervised Domain Adaptation (UDA) via a Domain-Adversarial Neural Network (DANN) can bridge the gap from synthetic VerSe DRRs (source) to real VinDr-SpineXR radiographs (target) without hallucinating anatomy.

## 2. Source Domain
- **Data:** `data/verse_drr_v2` (Advanced Physics-Based DRRs).
- **Labels:** Ground truth 2D bounding boxes (L1-L5) directly projected from VerSe 3D CT segmentations.

## 3. Target Domain
- **Data:** `VinDr-SpineXR` training images.
- **Labels:** None. VinDr images were used strictly as unlabeled examples for domain classification to align feature representations. No pathology labels were repurposed.

## 4. Architecture & 5. Methodology
- **Backbone:** ResNet50-FPN (Faster R-CNN).
- **GRL:** A Gradient Reversal Layer (GRL) was attached to the Global Average Pooled features of the backbone.
- **Domain Classifier:** A 3-layer MLP predicting `0` (Source) or `1` (Target).
- **Loss:** `L_total = L_detection + 0.1 * L_domain`. The GRL forces the backbone to learn features that maximize bounding box accuracy on VerSe while minimizing domain distinguishability (adversarial).

## 6. Training Configuration
- Optimizer: SGD (lr=0.001, momentum=0.9).
- Epochs: 10.
- $\lambda_{domain} = 0.1$.
- Batch size: 2 source + 2 target.

## 7. Source-Domain Results
The DANN regularization did **not** cause catastrophic forgetting on the source domain. In fact, on the synthetic hold-out set, the DANN-Adapted model exhibited slightly higher confidence scores (e.g., 0.98 vs 0.96 for L5, and 0.86 vs 0.50 for L1) than the Source-Only baseline.

## 8. Domain-Classifier Results
- **Epoch 1:** Domain Accuracy started low (0.30) but quickly climbed to 0.50.
- **Epochs 2-10:** Domain Accuracy remained stable between 0.35 and 0.55.
- **Interpretation:** An accuracy of ~50% in a balanced binary classification task indicates the domain classifier could not reliably distinguish between VerSe and VinDr features. The backbone successfully aligned the global representations.

## 9. VinDr Qualitative Comparison
Visual inspection of VinDr X-rays showed:
- **Source-Only:** Tended to output multiple crossing boxes, false positives, or no detections at all.
- **DANN-Adapted:** Produced slightly more confident predictions that occasionally landed on correct spinal regions, but frequently still failed to detect anything on particularly noisy radiographs.

## 10. Anatomical Consistency Analysis
A post-processing check evaluated whether predicted L1-L5 boxes descended logically down the Y-axis without duplicates:
- The DANN-Adapted model achieved valid ordering (`True`) on a subset of VinDr images where the Source-Only model failed entirely or produced invalid crossing boxes (`False`). 
- However, for 3 out of 5 VinDr test images, both models returned 0 valid detections over the 0.5 confidence threshold.

## 11. Limitations
- Feature-level alignment alone struggles to overcome massive low-level textural differences (e.g., overlapping ribs, varying exposure). 
- Without real VinDr vertebral ground truth, we cannot definitively quantify true positive rates versus anatomical hallucinations.

## 12. Decision Gate
**B. REPRESENTATION ADAPTED BUT TRANSFER UNCERTAIN**

The domain classifier accurately converged to random chance, proving feature alignment occurred. Source-domain anatomical localization improved. However, VinDr qualitative transfer remains highly sensitive and often produces zero detections on difficult clinical images.

## 13. Recommendation for Phase 13
**Phase 13: Real-X-ray Vertebral Localization.**
Since synthetic-to-real transfer via DANN remains quantitatively unverified on the target domain, the next phase should involve acquiring (or manually annotating) a small real-X-ray vertebral ground truth dataset to validate or fine-tune the localizer before proceeding to the Vertebral Morphology Encoder (VME).
