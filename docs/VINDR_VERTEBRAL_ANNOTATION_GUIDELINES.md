# VinDr-SpineXR Vertebral Annotation Guidelines

These guidelines define the protocol for manually creating vertebral ground truth annotations for 30-50 real VinDr-SpineXR images. The resulting dataset will be used strictly to validate and fine-tune the anatomical localizer.

## 1. Vertebral Body Boundary
- **Include:** The main vertebral body (corpus vertebrae). The bounding box should tightly encompass the superior and inferior endplates, and the anterior and posterior cortical margins of the vertebral body.
- **Exclude:** Posterior elements (spinous processes, transverse processes, facet joints, pedicles) unless they overlap the 2D projection of the main body in AP/Lateral views.

## 2. Vertebral Identification (L1–L5)
- Vertebrae must be labeled L1, L2, L3, L4, or L5 based on anatomical sequence (counting up from the sacrum or down from T12/ribs).
- **DO NOT** infer vertebral identity solely from VinDr pathology labels. Use visual anatomical position.

## 3. Visibility and Uncertainty
- **Visible:** The vertebral body margins are clearly visible.
- **Partially Visible:** The vertebra is at the edge of the crop, or severely obscured by artifacts/bowel gas, but still confidently identifiable.
- **Not Visible / Unidentifiable:** If a vertebra cannot be confidently identified or is completely cropped out, DO NOT annotate it. Do not force an L1-L5 label if you are unsure.

## 4. Handling Overlaps & Pathology
- **Overlapping Vertebrae:** In severely scoliotic or deformed spines, draw the box around the full visible extent of the vertebral body, even if it overlaps another bounding box.
- **Severe Pathology/Deformation:** Include wedge fractures, osteophytes, and collapsed vertebrae within the bounding box.
- **Surgical Implants:** Include surgical hardware (screws/plates) ONLY if they are embedded within the vertebral body itself.

## 5. Cropped Anatomy
- If the image cuts off the spine midway through a vertebra, draw the box only around the visible portion of that vertebra and mark it as `partially_visible`.

## 6. Annotation Format
Annotations will be saved as JSON:
```json
{
    "image_id": "...",
    "annotations": [
        {
            "vertebra": "L1",
            "bbox": [x_min, y_min, x_max, y_max],
            "visibility": "visible"
        }
    ]
}
```
