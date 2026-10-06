# Proposed NeuroMorphNet Data Requirements

This document outlines the strict data requirements for implementing the proposed architecture: **Anatomy-Constrained Dual-Graph NeuroMorphNet for Explainable Multi-Pathology Analysis of Spine Images**.

## 1. Required Supervisions
To fully implement the proposed components, the following supervisions are required:

### A. Vertebral Anatomy Supervision (For VME & ARG)
- **Requirement:** Localization (bounding boxes or centroids) and Identification (e.g., L1, L2, L3) of all visible vertebrae in the image.
- **Ideal format:** Instance segmentation masks for each vertebra.
- **Current Status:** **MISSING** in VinDr-SpineXR. Available in VerSe (3D CT).

### B. Pathology Supervision (For Multi-Pathology Classification)
- **Requirement:** Image-level binary flags for the presence of specific spinal pathologies (e.g., Osteophytes, Spondylolysthesis, Foraminal Stenosis).
- **Current Status:** **AVAILABLE** in VinDr-SpineXR.

### C. Pathology Spatial Supervision (For Object Detection / Optional 2nd Graph)
- **Requirement:** Bounding boxes identifying the location of the pathologies within the image.
- **Current Status:** **AVAILABLE** in VinDr-SpineXR (but must be used carefully to avoid annotation leakage during inference).

### D. Severity Supervision (For Severity Grading)
- **Requirement:** Ordinal or categorical labels indicating the severity of the identified pathologies (e.g., Normal, Mild, Moderate, Severe).
- **Current Status:** **MISSING** in both VinDr-SpineXR and VerSe.

## 2. Component Data Dependencies

| Component | Depends On | Available Data Source |
|---|---|---|
| **Vertebral Morphology Encoder (VME)** | Vertebral bounds/masks | None directly (Requires Transfer Learning) |
| **Anatomical Relation Graph (ARG)** | Vertebral identity & bounds | None directly (Requires Transfer Learning) |
| **Morphology-Aware Cross-Attention (MCAM)** | VME, ARG, Global Image | Algorithmic (Requires VME/ARG) |
| **Uncertainty-Guided Decision Module (UGDM)** | Model probabilities | Algorithmic |
| **Multi-Pathology Classifier** | Image-level labels | VinDr-SpineXR |

## 3. The Modality Transfer Challenge
Because VinDr-SpineXR consists of 2D radiographs and lacks anatomical annotations, and VerSe consists of 3D CTs with full anatomical annotations, the primary data engineering challenge for this project is **Cross-Modality Anatomy Localization**.

A pipeline must be built to transfer anatomical knowledge from 3D CTs (or 2D projections of those CTs) to the 2D clinical X-rays to generate the necessary pseudo-labels required to build the VME and ARG.
