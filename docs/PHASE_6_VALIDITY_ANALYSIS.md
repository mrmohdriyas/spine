# Phase 6 — Validity Analysis

## 1. Purpose
The purpose of this analysis is to evaluate whether the large performance improvements seen in NeuroMorphNet V1 (Micro F1 0.8689) compared to the ResNet18 Baseline (Micro F1 0.4000) are due to legitimate morphological representation learning, or whether there is **information leakage** from ground-truth pathology bounding boxes. Additionally, we analyze why the NeuroMorphNet V2 Spatial Relation Graph (Micro F1 0.6812) degraded performance relative to V1.

## 2. Morphology Feature Analysis
An analysis of the morphology features across the training set revealed strong correlations between the geometric properties of bounding boxes and specific pathology classes.
- **Surgical implants** are strongly associated with large areas and tall heights (e.g., rods and screws spanning multiple vertebrae).
- **Osteophytes** are strongly associated with high annotation counts (averaging 4.09 per image) and specific spatial locations (edges of vertebrae).
- **No finding** images contain zero annotations by definition, meaning their morphology vector is always zero.

## 3. Potential Label Leakage
### Confirmed
Label leakage is present and strongly influences the V1 model's high performance. Because the morphology vector directly encodes the quantity and geometric properties of ground-truth pathology annotations, the model does not need to look at the image to predict certain classes. 
- The `num_annotations` feature strongly predicts `Osteophytes` (correlation 0.73). 
- A zero-vector perfectly predicts `No finding` (perfect negative correlation across all non-zero geometry features).
- The geometry (size/aspect ratio) strongly predicts `Surgical implant`.
The V1 model acts partially as a bounding-box classifier rather than purely an image classifier.

## 4. Spatial Graph Analysis
### Confirmed
The V2 Spatial Relation Graph only constructed edges for images containing 2 or more annotations (39.1% of the dataset). 
- 51.2% of the dataset (the `No finding` cases) produced 0 nodes.
- 9.6% of the dataset produced 1 node.
As a result, the spatial graph branch provided no structural information for over 60% of the dataset, producing zero-vectors or isolated node projections.

## 5. V1 vs V2 Per-Class Comparison

| Pathology Class | V1 F1 | V2 F1 | Difference |
| --- | ---: | ---: | ---: |
| Disc space narrowing | 0.44 | 0.25 | -0.19 |
| Foraminal stenosis | 0.33 | 0.00 | -0.33 |
| Osteophytes | 0.98 | 0.88 | -0.10 |
| Spondylolysthesis | 0.33 | 0.50 | +0.17 |
| Surgical implant | 1.00 | 1.00 | 0.00 |
| Vertebral collapse | 0.00 | 0.00 | 0.00 |
| Other lesions | 0.67 | 0.10 | -0.57 |
| No finding | 1.00 | 0.96 | -0.04 |

The classes that suffered the largest drops (`Other lesions`, `Foraminal stenosis`, `Disc space narrowing`) are classes with few annotations or highly variable geometry.

## 6. Observed Reasons for V2 Performance Change
### Confirmed
- **Redundant Information:** The Spatial Graph Encoder aggregates node geometry (via Mean Pooling) after message passing. This is largely redundant with the V1 Morphology Encoder, which already computes `mean_w`, `mean_h`, `mean_area`, etc.
- **Overfitting:** The V2 architecture achieved a lower training loss (0.1006) than V1 (0.1043) but a significantly higher validation loss (0.9560 vs 0.5975). The increased capacity and redundant parameters led to overfitting.

### Suspected
- **Missing Node Identity:** The graph nodes only contain box geometry (6D), lacking visual features (ROI pooling) or anatomical semantic identity (e.g., "L4 Vertebra"). Passing purely geometric messages between anonymous pathology boxes likely creates noisy representations that confuse the classifier rather than helping it.

## 7. Dataset Limitations
### Confirmed
- The dataset lacks vertebral segmentations or masks. We cannot build a true anatomical graph without these.
- The test set is extremely small (48 valid images). While the V1 and V2 F1 differences are large, the absolute performance on minority classes is highly sensitive to a single test sample shifting from True Positive to False Negative.
- Providing ground-truth pathology bounding boxes as input features during inference constitutes an invalid evaluation setup for a real-world clinical application, as those boxes would not be available before prediction.

## 8. Recommended Final Architecture
### Confirmed
Because V1 and V2 suffer from ground-truth label leakage via the morphology vector, neither is a valid architecture for real-world classification without relying on an upstream pathology detection model. 

The **ResNet18 Baseline** remains the only valid architecture tested so far that predicts strictly from the image without leaking ground-truth annotations.

## 9. Recommended Next Experiment
To correctly utilize the bounding box annotations without leaking labels, the architecture should be converted from an Image Classification model into an **Object Detection** model (e.g., Faster R-CNN or YOLO). This allows the network to predict the bounding boxes and their classifications directly from the image, using the current morphology data exclusively as training targets rather than input features.
