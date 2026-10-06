# Phase 1 — Dataset Audit & Understanding

## 1. Dataset Location
`C:\Users\Mohamed Riyas\.cache\kagglehub\datasets\siddhale937e92739\annotated-medical-image-dataset-for-spinal-lesions\versions\1\physionet.org\files\vindr-spinexr\1.0.0`

## 2. Dataset Size
Approximate size: ~9.91 GB
Total files: 3,155 files (including annotations, HTML, txt, dicom).

## 3. File Structure
The dataset contains a subset of `physionet.org` specifically `vindr-spinexr/1.0.0`.
- `annotations/` (contains `train.csv`, `test.csv` and `index.html`)
- `train_images/` (DICOM images)
- `test_images/` (DICOM images)
- Assorted metadata files: `index.html`, `LICENSE.txt`, `SHA256SUMS.txt`, `supplemental_file_DICOM_tags_SpineXR.pdf`

Total files by extension:
- `.dicom`: 3,146
- `.html`: 4
- `.csv`: 2
- `.txt`: 2
- `.pdf`: 1

## 4. Image Information
- **Image modality**: X-Ray (implied by SpineXR)
- **Image format**: DICOM (`.dicom`)
- **Dimensions**: Variable sizes depending on the sample (e.g., 946x1420, 1142x2874, 3000x3000, 1718x1939, 1025x1581, etc.)
- **Number of channels**: 1 (Grayscale)
- **Number of images**: 3,146 total DICOM images
- **Dimensions consistency**: Images have different dimensions.

## 5. Annotation Information
- **Formats**: CSV (`train.csv`, `test.csv`)
- **Type**: Bounding boxes (`xmin`, `ymin`, `xmax`, `ymax`) with an associated `lesion_type`.

## 6. Available Labels
- **Source**: `annotations/train.csv` (column `lesion_type`)
- **Number of classes**: 8
- **Class names**: Osteophytes, No finding, Disc space narrowing, Other lesions, Surgical implant, Foraminal stenosis, Spondylolysthesis, Vertebral collapse.
- **Type**: Bounding box localization of pathologies.

## 7. Class Distribution
| Class | Number of samples | Percentage |
| :--- | :--- | :--- |
| Osteophytes | 12,556 | 64.22% |
| No finding | 4,260 | 21.79% |
| Disc space narrowing | 924 | 4.73% |
| Other lesions | 446 | 2.28% |
| Surgical implant | 429 | 2.19% |
| Foraminal stenosis | 387 | 1.98% |
| Spondylolysthesis | 280 | 1.43% |
| Vertebral collapse | 268 | 1.37% |

*Note: The dataset exhibits severe class imbalance. "Osteophytes" accounts for the majority of annotations, while serious pathologies like "Vertebral collapse" and "Spondylolysthesis" are scarce.*

## 8. Vertebral Information
- **Vertebral localization**: Not available explicitly (annotations are for lesion bounding boxes, not specific vertebral bodies like L4/L5).
- **Vertebral masks**: Not available.
- **Vertebral IDs**: Not available.
- **Vertebral-level pathology**: Unclear/Not available. (Pathologies are annotated with bounding boxes, but there are no specific IDs tying the box to a specific vertebra).

## 9. Dataset Quality Issues
- Severe class imbalance (64% of samples are a single class).
- Varying image dimensions, which will require aggressive resizing or padding during pre-processing.
- High number of images marked as "No finding" but taking up bounding box definitions (likely image-level).
- Bounding boxes are provided, but pixel-level segmentation masks are absent.

## 10. Representative Samples
A sample visualization has been generated and saved locally to: `docs/sample_visualization.png`. 
It displays the DICOM image along with the bounding boxes and their corresponding labels.

## 11. NeuroMorphNet Feasibility
NeuroMorphNet aims to be a "Morphology-Aware Graph Network for Explainable Spine Pathology Analysis".
Since graph networks typically require nodes (e.g., individual vertebrae identified and localized) and edges (anatomical relationships), building an anatomical graph will be very challenging without vertebral IDs or segmentation masks.

## 12. Limitations
The primary limitation is the lack of anatomical context (vertebral IDs and masks) required for structural graph construction.

---

## Phase 1 Conclusion

### What the dataset supports
- Object detection for spinal pathologies (bounding box prediction).
- Image-level classification of spine lesions.

### What the dataset does not support
- Vertebra-specific identification (e.g., distinguishing L4 from L5).
- Pixel-level semantic segmentation (masks).
- Direct extraction of morphological measurements (e.g., vertebral height or angle) without an intermediate segmentation step.
- Out-of-the-box anatomical graph construction.

### Recommended task for NeuroMorphNet
- Object Detection for lesion localization, potentially paired with a heuristic to estimate vertebral locations, OR switching to an object detection architecture rather than a pure GNN.
- Alternatively, predicting pathologies globally per image rather than per node in a graph.

### Required assumptions
- If continuing with a GNN, we must assume that an off-the-shelf vertebral segmentation/localization model (or another dataset) will be needed to first extract the vertebrae to form the graph nodes, as this dataset only provides lesion boxes.

### Phase 2 Recommendation
- Instead of building a GNN from scratch using non-existent nodes, we should pivot Phase 2 to **Object Detection** (e.g., YOLO, Faster R-CNN) or **Classification** using the bounding boxes provided. 
- If a Graph Network is strictly required, Phase 2 must involve implementing or borrowing a pre-trained vertebral localization model to generate the graph nodes automatically before building the pathology prediction network.
