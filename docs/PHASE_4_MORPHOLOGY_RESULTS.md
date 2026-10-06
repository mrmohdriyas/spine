# Phase 4 — Morphology-Aware CNN Results

## NeuroMorphNet V1 Architecture
The baseline ResNet18 model was extended to include morphological/geometric features derived from the **annotated pathology-region bounding boxes**. 

- **Image Stream**: Image → ResNet18 → Image Features (512D)
- **Morphology Stream**: Bounding Box Geometries (mean width, height, area, aspect ratio, center x/y, max area, annotation count) → MLP → Morphology Features (64D)
- **Feature Fusion**: Concatenation of the two streams (576D) passed through a fusion layer to produce the 8 independent pathology logits.

*Note: The morphological features represent pathology-region geometry, not vertebral anatomy.*

## Dataset & Training
- **Train**: 2633 images
- **Validation**: 464 images
- **Test**: 49 images (48 valid test images evaluated after excluding corrupted DICOMs)
- **Invalid DICOMs**: 2 (Documented in Phase 3)
- **Task**: 8-class multi-label pathology classification
- **Loss**: `BCEWithLogitsLoss`
- **Training Epochs**: 10
- **Final Train Loss**: 0.1043
- **Final Validation Loss**: 0.5975

## Results & Comparison

Incorporating the geometric properties of the pathology annotations significantly improved all evaluation metrics compared to the standard CNN baseline.

| Model | Micro F1 | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| ResNet18 Baseline | 0.4000 | 0.2751 | 0.5970 |
| **NeuroMorphNet V1 (ResNet18 + Morphology)** | **0.8689** | **0.5946** | **0.9220** |

> **Limitations:** The test set is small, so the resulting test metrics should be interpreted cautiously and should not be treated as strong evidence of clinical generalization.

## Checkpoint & Logs
- Checkpoint: `outputs/neuromorphnet_v1/best_model.pt`
- Classification Report: `outputs/neuromorphnet_v1/classification_report.csv`
- Confusion Matrices: `outputs/neuromorphnet_v1/confusion_matrix.png`
