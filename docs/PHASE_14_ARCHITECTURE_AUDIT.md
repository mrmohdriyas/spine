# Phase 14 Architecture Audit

Before constructing the final Anatomy-Constrained Dual-Graph NeuroMorphNet, we have audited the repository to identify reusable components from previous phases. This prevents unnecessary duplication of functionality.

## 1. Reusable Data Infrastructure
- **VinDr Dataset Loader:** `src/data/dataset.py` contains the established `VinDrDataset` class which correctly parses DICOMs and loads the 8 multi-label pathologies.
- **Image Preprocessing:** Included in `src/data/dataset.py` (ResNet normalization, 512x512 resizing).
- **Class Mapping:** Established multi-label indices for the 8 target pathologies.
- **Manifests:** `data/manifests/train.csv` and `data/manifests/test.csv`.

## 2. Reusable Anatomical Data (Phase 13)
- **Real-X-ray Annotations:** `data/vindr_vertebral_annotations/train.json`, `val.json`, and `test.json` containing the (flawed but structurally correct) human annotations for L1-L5.

## 3. Reusable Architectures & Checkpoints
- **Baseline Model:** `src/models/resnet_baseline.py` and `outputs/baseline/best_model.pt`.
- **NeuroMorphNet V3 Prototype:** `src/models/neuromorphnet_v3.py` established the image-derived morphology extractor without geometry leakage. This paradigm will be used for the final VME.
- **DANN Localizer:** `scripts/models/dann_faster_rcnn.py` and `outputs/phase13_real_xray_validation/detector_dann_finetuned.pt` will serve as the base for the Vertebral Localizer, requiring supervised fine-tuning on the real X-rays.

## 4. Reusable Analysis Tools
- **Explainability (Grad-CAM):** Logic established in Phase 8 (`scripts/gradcam_analysis.py`).
- **Leakage Prevention:** Concepts established in Phase 7 to aggressively strip bounding boxes from model inputs.

## 5. Components to Build from Scratch (Phase 14)
1. `vertebral_localizer.py`
2. `vertebral_morphology_encoder.py` (VME)
3. `anatomical_relation_graph.py` (ARG)
4. `pathology_relation_graph.py`
5. `dual_graph_fusion.py`
6. `mcam.py` (Morphology-Aware Cross-Attention Module)
7. `pathology_head.py`
8. `uncertainty_module.py` (UGDM)
9. `neuromorphnet_final.py`
