# Phase 3 — Simple CNN Baseline Results

## Dataset Setup
VinDr-SpineXR is treated as a **multi-label pathology classification problem** because individual images may contain multiple pathology annotations. Because of this, `BCEWithLogitsLoss` is used instead of `CrossEntropyLoss`. The original annotations were aggregated by `image_id` to produce 8-dimensional multi-hot target vectors.

- **Train**: 2633 images
- **Validation**: 464 images
- **Test**: 49 images (48 valid images after integrity filtering)
- **Classes**: 8

## Data Integrity
During testing and validation, a dataset integrity issue was discovered. A lightweight integrity check across the entire dataset revealed:
- **Total images checked**: 3146
- **Valid images**: 3144
- **Invalid images**: 2 (Missing Pixel Data)

The corrupted DICOM files are documented in `docs/invalid_dicom_files.csv` and are explicitly excluded from the dataset during evaluation to prevent crashes and metric corruption.

## Model
- **Architecture**: ResNet18
- **Classifier**: 8-class classifier (8 independent logits)

## Hardware
- **Device**: NVIDIA GeForce RTX 4050 Laptop GPU
- **Compute API**: CUDA

## Training configuration
- **Batch size**: 8
- **Learning rate**: 1e-4
- **Epochs**: 10
- **Optimizer**: AdamW
- **Loss**: `BCEWithLogitsLoss` (with `pos_weight` calculated from training set to handle class imbalance)
- **Random seed**: 42

## Training Behavior
The model was successfully trained for 10 epochs. 
- **Final Train Loss**: 0.0866
- **Final Validation Loss**: 1.6016

The substantial difference between the final train loss (0.0866) and the validation loss (1.6016) suggests that the model may be overfitting on the training dataset.

## Evaluation Results

The evaluation was performed using the valid test split containing 48 images. The model predictions were obtained via `sigmoid(logits)` with a binary threshold of 0.5.

> **Limitations:** The test set is small, so the resulting test metrics should be interpreted cautiously and should not be treated as strong evidence of clinical generalization.

| Metric      | Test Set |
| ----------- | -------: |
| Micro F1    | 0.4000   |
| Macro F1    | 0.2751   |
| Weighted F1 | 0.5970   |

*Refer to `outputs/baseline/classification_report.csv` for per-class details, including Precision, Recall, and F1 scores.*
*The per-class multi-label confusion matrices are visualized in `outputs/baseline/confusion_matrix.png`.*
