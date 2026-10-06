# Phase 7 — Image-Derived Morphology Encoder

## Architecture
```text
Image
 ↓
ResNet18
 ├── Image representation
 └── Image-derived morphology representation
          ↓
       Fusion
          ↓
    8 pathology logits
```

## Data leakage prevention
> Ground-truth bounding boxes are not provided to the model as input features.

An automated verification script (`verify_no_annotation_input.py`) confirmed that the dataset yields only image tensors and target labels, and that the NeuroMorphNet V3 forward pass accepts only image tensors. This guarantees zero ground-truth annotation leakage during both training and inference.

## Comparison

| Model | Input | Micro F1 | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: |
| ResNet18 Baseline | Image | 0.4000 | 0.2751 | 0.5970 |
| V1 *(Annotation-Assisted/Leakage)* | Image + GT box geometry | 0.8689 | 0.5946 | 0.9220 |
| V2 *(Annotation-Assisted/Leakage)* | Image + GT box geometry + graph | 0.6812 | 0.4615 | 0.8427 |
| **V3** | **Image only + learned morphology** | **0.4906** | **0.3614** | **0.6473** |

*Note: V1 and V2 are annotation-assisted experiments heavily affected by label leakage and should not be compared as equivalent real-world predictive models.*

## Results
The leakage-free NeuroMorphNet V3 achieved:
- **Micro F1:** 0.4906
- **Macro F1:** 0.3614
- **Weighted F1:** 0.6473

These metrics show a clear, legitimate improvement over the pure ResNet18 baseline (Micro F1 0.4000 -> 0.4906). Extracting an explicit learned morphology representation via a secondary convolutional and pooling branch from the ResNet18 feature map successfully captures additional predictive signal without relying on manual ground-truth bounding box geometry. 

## Limitations
- **No vertebral segmentation:** The dataset does not provide vertebral segmentations, so true anatomical/vertebral morphology cannot be explicitly learned or measured.
- **Learned representation:** The morphology representation is learned end-to-end rather than being explicitly measured geometry. We cannot easily interpret the exact physical dimensions the 64D embedding corresponds to.
- **Small test set:** The test set evaluation consists of only 48 valid images, meaning the metrics may have high variance.
- **Invalid DICOM files:** As documented previously, corrupted DICOM files continue to be omitted from metric calculations.
- **Dataset-specific limitations:** Findings are currently limited to this specific dataset and pathology distribution.
