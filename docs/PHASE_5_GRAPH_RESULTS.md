# Phase 5 — Spatial Relation Graph Results

## Architecture
```text
Image
 ↓
ResNet18
 ↓
Image Features (512D)
 +
Morphology Encoder (64D)
 +
Spatial Relation Graph (64D)
 ↓
Feature Fusion
 ↓
8 pathology logits
```

## Dataset
Train: 2633 images
Validation: 464 images
Test: 49 images
Valid test images: 48
Invalid DICOMs: 2 (Documented in Phase 3/4)

## Training
Epochs: 10

V1 Train Loss: 0.1043
V1 Validation Loss: 0.5975

V2 Train Loss: 0.1006
V2 Validation Loss: 0.9560

## Results

Adding a Spatial Relation Graph to the NeuroMorphNet V1 architecture resulted in a noticeable drop in performance across all metrics on the test set, while simultaneously increasing the validation loss significantly over V1. This suggests that the simple spatial connections between pathology-region bounding boxes might be adding noise or promoting overfitting rather than providing useful auxiliary context. 

| Model | Micro F1 | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| ResNet18 Baseline | 0.4000 | 0.2751 | 0.5970 |
| NeuroMorphNet V1 (ResNet18 + Morphology) | **0.8689** | **0.5946** | **0.9220** |
| NeuroMorphNet V2 (+ Spatial Graph) | 0.6812 | 0.4615 | 0.8427 |

### Per-Class F1 Comparison (V1 vs V2)
*(Note: F1 scores extracted from V1 and V2 classification reports)*

| Pathology Class | V1 F1 | V2 F1 |
| --- | ---: | ---: |
| Disc space narrowing | 0.44 | 0.25 |
| Foraminal stenosis | 0.33 | 0.00 |
| Osteophytes | 0.98 | 0.88 |
| Spondylolysthesis | 0.33 | 0.50 |
| Surgical implant | 1.00 | 1.00 |
| Vertebral collapse | 0.00 | 0.00 |
| Other lesions | 0.67 | 0.10 |
| No finding | 1.00 | 0.96 |

## Graph Contribution
The primary question was whether the spatial relation graph provides additional predictive information beyond the morphology-aware CNN. 

The measured differences are:
- **Micro F1 difference:** -0.1877
- **Macro F1 difference:** -0.1331
- **Weighted F1 difference:** -0.0793

**Interpretation:** The spatial arrangement of annotated pathology regions did *not* contribute additional predictive information in this experimental setup. Instead, the addition of the Spatial Relation Graph severely degraded generalization performance. The V2 model was able to achieve a slightly lower training loss than V1 (0.1006 vs 0.1043) but suffered a much higher validation loss (0.9560 vs 0.5975), indicating that the spatial graph exacerbated overfitting.

## Limitations
1. The graph nodes represent annotated pathology regions.
2. The graph is not a vertebral anatomical graph.
3. The dataset has no vertebral IDs or vertebral masks.
4. Morphology and graph features are derived from annotated pathology bounding boxes.
5. The test set is small, meaning these metric shifts, while large, apply to a small sample size.
6. Two invalid DICOM files were identified and excluded from the test set evaluation.
