# Phase 8 — NeuroMorphNet V3 Explainability & Error Analysis

## 1. Objective
The objective of this phase is to analyze what the leakage-free NeuroMorphNet V3 model is learning. This involves examining per-class performance, understanding where V3 improves over the ResNet18 baseline, and providing interpretable visual evidence (Grad-CAM) and quantitative analysis of the image-derived morphology embedding.

## 2. V3 Architecture Summary
```text
Image
 ↓
ResNet18 feature map
 ├── image representation
 └── morphology encoder
        ↓
      morphology embedding (64D)
        ↓
      fusion
        ↓
      pathology logits
```

## 3. Test-set Composition
- **Total Valid Test Images**: 48
- Invalid DICOMs were excluded as established in previous phases.
- The same test split was used for both the ResNet18 Baseline and NeuroMorphNet V3.

## 4. Per-Class Metrics
*(Metrics are calculated on the test set using a 0.5 probability threshold)*

| Pathology | Precision | Recall | F1-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| No finding | 0.74 | 0.61 | 0.67 | 23 |
| Disc space narrowing | 0.13 | 1.00 | 0.24 | 2 |
| Foraminal stenosis | 0.00 | 0.00 | 0.00 | 1 |
| Osteophytes | 0.68 | 0.79 | 0.73 | 24 |
| Other lesions | 0.07 | 1.00 | 0.12 | 1 |
| Spondylolysthesis | 0.07 | 1.00 | 0.13 | 1 |
| Surgical implant | 1.00 | 1.00 | 1.00 | 2 |
| Vertebral collapse | 0.00 | 0.00 | 0.00 | 1 |

**Overall Metrics:**
- **Micro F1:** 0.4906
- **Macro F1:** 0.3614
- **Weighted F1:** 0.6473

*Note: Classes like Foraminal stenosis, Vertebral collapse, Spondylolysthesis, and Other lesions have only 1 positive example in the test set. Results for these classes should be interpreted with extreme caution due to the small sample size.*

## 5. Confusion Matrices
Binary confusion matrices for each class have been generated and saved to:
`outputs/neuromorphnet_v3/explainability/confusion_matrices/combined_matrices.png`

## 6. Baseline vs V3 Comparison

| Pathology | Baseline F1 | V3 F1 | Difference |
| --- | ---: | ---: | ---: |
| No finding | 0.60 | 0.67 | +0.06 |
| Disc space narrowing | 0.19 | 0.24 | +0.04 |
| Foraminal stenosis | 0.13 | 0.00 | -0.13 |
| Osteophytes | 0.73 | 0.73 | 0.00 |
| Other lesions | 0.09 | 0.12 | +0.03 |
| Spondylolysthesis | 0.12 | 0.13 | +0.01 |
| Surgical implant | 0.33 | 1.00 | +0.67 |
| Vertebral collapse | 0.00 | 0.00 | 0.00 |

### Detailed Error Counts

| Pathology | Baseline TP | V3 TP | Baseline FP | V3 FP | Baseline FN | V3 FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No finding | 13 | 14 | 7 | 5 | 10 | 9 |
| Disc space narrowing | 2 | 2 | 17 | 13 | 0 | 0 |
| Foraminal stenosis | 1 | 0 | 13 | 8 | 0 | 1 |
| Osteophytes | 19 | 19 | 9 | 9 | 5 | 5 |
| Other lesions | 1 | 1 | 20 | 14 | 0 | 0 |
| Spondylolysthesis | 1 | 1 | 15 | 13 | 0 | 0 |
| Surgical implant | 2 | 2 | 8 | 0 | 0 | 0 |
| Vertebral collapse | 0 | 0 | 12 | 3 | 1 | 1 |

## 7. Error Analysis
V3 generally maintained the same True Positive (TP) count as the Baseline while significantly reducing the number of False Positives (FPs). 
- **Surgical implant:** V3 achieved perfect F1 (1.00) by eliminating all 8 false positives made by the Baseline.
- **No finding:** V3 produced fewer false negatives and fewer false positives in this test set.
- **Vertebral collapse & Other lesions:** V3 significantly reduced false positives (e.g., dropping from 12 to 3 for Vertebral collapse), indicating that the image-derived morphology branch helps the model better reject incorrect bounding-box-like features that confused the baseline model.
- **Foraminal stenosis:** V3 missed the only positive example in the test set, creating a false negative where the baseline produced a true positive.

Detailed error cases (TP, TN, FP, FN) with probabilities have been collected in `outputs/neuromorphnet_v3/explainability/error_cases.csv`.

## 8. Grad-CAM Methodology
Grad-CAM was implemented on the final convolutional feature layer of the ResNet18 backbone (prior to the split into the Image Representation and Morphology Encoder branches). The gradients were computed with respect to the specific pathology logit being analyzed.

## 9. Grad-CAM Examples
Grad-CAM heatmaps overlaid on original images for True Positives, False Positives, and False Negatives (where available) have been saved to:
`outputs/neuromorphnet_v3/explainability/gradcam/`

Grad-CAM provides qualitative evidence of image regions contributing to the prediction. For example, True Positive activations for Surgical Implants highlight dense, hardware-like structures in the spine. 

## 10. Morphology Embedding Analysis
The 64-dimensional learned morphology embedding was reduced to 2 dimensions using Principal Component Analysis (PCA) for visualization. 
- The resulting visualization is saved at: `outputs/neuromorphnet_v3/explainability/morphology_embedding_pca.png`.
- The embedding is an abstract learned representation. Individual dimensions do not map linearly to specific anatomical measurements, but the PCA indicates that the features form distinct clusters corresponding to the presence or absence of major pathologies.

## 11. Leakage Verification
The verification script (`scripts/verify_no_annotation_input.py`) was re-run.
- **Result:** PASS
- No bounding boxes, annotation counts, pathology coordinates, or ground-truth morphology vectors enter the model. The dataset provides only the image tensor and the multi-hot target vector.

## 12. Limitations
1. **Small Test Set:** The test set consists of only 48 images. Several classes have only 1 or 2 positive examples, meaning metrics like F1 score are highly volatile and heavily influenced by single predictions.
2. **Abstract Morphology:** V3 learned an image-derived morphology representation, but without ground-truth vertebral segmentations to enforce structural constraints, we cannot confirm that the representation strictly encodes anatomical geometry.
3. **Multi-label Complexity:** The model struggles to perfectly disentangle co-occurring pathologies (e.g., Osteophytes and Disc Space Narrowing).

## 13. Scientific Interpretation
The addition of an image-derived morphology branch (V3) provides a measurable improvement over a standard global average-pooled ResNet18 (Baseline). The error analysis suggests that this branch primarily acts as a **regularizer against false positives**. By forcing the network to maintain a spatial-morphological feature projection before fusion, the model is better equipped to reject ambiguous textural features that the Baseline mistakenly classified as pathologies (e.g., surgical implants).

## 14. Conclusion
NeuroMorphNet V3 successfully demonstrates that morphological representations can be learned directly from the image without relying on leaky ground-truth annotations. While its performance (Micro F1 0.4906) is lower than the annotation-leaking V1 model, it is a valid, leakage-free architecture that outperforms the image-only Baseline (Micro F1 0.4000). Future work should focus on acquiring proper vertebral segmentation masks to provide explicit anatomical supervision.
