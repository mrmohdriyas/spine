# Phase 9 — Proposed Algorithm Feasibility & Dataset Audit

## 1. Objective
The objective of this phase is to determine whether the currently available datasets (VinDr-SpineXR) and secondary datasets (VerSe) can support the proposed architecture: **"Anatomy-Constrained Dual-Graph NeuroMorphNet for Explainable Multi-Pathology Analysis of Spine Images"** without fabricated labels, annotation leakage, or unsupported anatomical assumptions.

## 2. Dataset 1 — VinDr-SpineXR
VinDr-SpineXR is a 2D radiograph (X-ray) dataset focused on pathology detection.
- **Image:** AVAILABLE (2D Radiographs)
- **Pathology labels:** AVAILABLE (Multi-label)
- **Pathology bounding boxes:** AVAILABLE
- **Vertebral localization:** NOT AVAILABLE
- **Vertebral identity:** NOT AVAILABLE
- **Vertebral segmentation:** NOT AVAILABLE
- **Vertebral morphology:** NOT AVAILABLE (Cannot be safely derived without localization/segmentation)
- **Pathology severity:** NOT AVAILABLE
- **Anatomical level:** NOT AVAILABLE (Pathology boxes do not specify which vertebra is affected)

## 3. Dataset 2 — VerSe (Large Scale Vertebrae Segmentation Challenge)
VerSe is a 3D CT dataset focused on anatomical vertebrae segmentation and identification.
- **Vertebral segmentation masks:** AVAILABLE
- **Vertebral centroids:** AVAILABLE
- **Vertebral identity:** AVAILABLE (C1-C7, T1-T12, L1-L5, transitional variants)
- **CT images:** AVAILABLE
- **Spacing information:** AVAILABLE

## 4. Architecture Feasibility

| Proposed Component | Required Data | Current VinDr | VerSe | Feasible? |
|---|---|---|---|---|
| **VME (Vertebral Morphology Encoder)** | Vertebral masks/boxes | No | Yes | **Partial** (Requires cross-modality transfer) |
| **ARG (Anatomical Relation Graph)** | Vertebral locations & identity | No | Yes | **Partial** (Requires cross-modality transfer) |
| **Dual Graph** | Two distinct graphical structures | No | Yes | **Partial** |
| **MCAM (Morphology-Aware Cross-Attention)** | VME features + Image features | No | N/A | **Yes** (If VME is built) |
| **UGDM (Uncertainty-Guided Decision Module)** | Probabilistic outputs / Ensembles | Yes | N/A | **Yes** (Algorithmic) |
| **Multi-pathology classification** | Multi-label pathology targets | Yes | No | **Yes** |
| **Explainability** | Attention weights / Grad-CAM | Yes | N/A | **Yes** (Algorithmic) |
| **Severity grading** | Severity labels (e.g., Mild/Severe) | No | No | **No** (Not safe to assume) |

## 5. Dual-Graph Definition Analysis
The term "Dual-Graph" requires explicit definition. We analyze two viable candidates:

### Definition A: Vertebral Anatomical Graph + Pathology-Instance Relation Graph
- **Required Nodes:** Vertebrae (Graph 1), Pathology Regions (Graph 2).
- **Required Edges:** Anatomical adjacency (Graph 1), Spatial proximity (Graph 2).
- **Available Supervision:** None directly in VinDr. Requires an upstream Vertebral Object Detector (trained on VerSe) and an upstream Pathology Object Detector (trained on VinDr).
- **Feasibility:** High, but modular.

### Definition B: Vertebral Anatomical Graph + Vertebral Morphology Similarity Graph
- **Required Nodes:** Vertebrae (Both Graphs).
- **Required Edges:** Anatomical adjacency (Graph 1), Morphological feature similarity (Graph 2).
- **Available Supervision:** Requires accurate vertebral segmentations to extract morphology features.
- **Feasibility:** Very challenging on VinDr 2D X-rays without an extremely robust cross-modality segmentation model.

## 6. Modality Compatibility
- **VinDr-SpineXR (2D X-ray)** vs **VerSe (3D CT)**
- They **cannot** be directly combined as paired data.
- They **cannot** be used in a single joint training batch without complex 2D/3D projection networks.
- They **can** support transfer learning: A segmentation model trained on 2D Digitally Reconstructed Radiographs (DRRs) generated from VerSe CTs could be applied to VinDr-SpineXR to generate pseudo-labels for vertebral localization.

## 7. Proposed End-to-End Pipeline
Because the datasets are not paired, the proposed pipeline must be highly modular:

```text
1. Image (2D X-ray)
        ↓
2. Vertebral Detection & Identification Module [Supervision: VerSe-derived pseudo-labels]
        ↓
3. Vertebral Morphology Encoder (VME) [Extracts features per vertebra]
        ↓
4. Anatomical Relation Graph (ARG) [Nodes: Vertebrae, Edges: Adjacency]
        ↓
5. Global Image Encoder (ResNet/ViT) [Supervision: VinDr]
        ↓
6. Pathology Region Detector (Optional 2nd Graph) [Supervision: VinDr bounding boxes]
        ↓
7. Morphology-Aware Cross-Attention Module (MCAM) [Fuses ARG + Global Features]
        ↓
8. Uncertainty-Guided Decision Module (UGDM)
        ↓
9. Multi-Pathology Classification [Supervision: VinDr]
        ↓
10. Explainability (Attention Maps / Grad-CAM)
```

## 8. Current Results (Reference)
- ResNet18: Micro F1 = 0.4000
- V3 (Clean Image-Derived): Micro F1 = 0.4906
- V1 (Annotation-Assisted/Leakage): Micro F1 = 0.8689
- V2 (Annotation-Assisted/Leakage): Micro F1 = 0.6812

## 9. Research Gap
To transition from V3 to the proposed architecture, the following must be developed:
1. **Vertebral Localization & Identification:** A model to find and label vertebrae in VinDr X-rays.
2. **Explicit Morphology Extraction:** A method to extract physical morphology from localized vertebrae.
3. **Anatomical Graph Construction:** Connecting the localized vertebrae.
4. **Cross-Attention & Uncertainty:** New architectural modules (MCAM, UGDM).
5. **Proper Evaluation:** Evaluating without relying on ground-truth boxes as inputs.
