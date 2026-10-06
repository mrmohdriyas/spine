import os
import torch
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.dann_faster_rcnn import DANN_FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_source_only_model():
    num_classes = 6
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(r"c:\projects\spine\outputs\phase11_domain_adaptation\detector_v2.pt", weights_only=True))
    model.eval()
    return model

def get_dann_model():
    model = DANN_FasterRCNN(num_classes=6)
    model.faster_rcnn.load_state_dict(torch.load(r"c:\projects\spine\outputs\phase12_domain_adaptation\detector_dann.pt", weights_only=True))
    model.eval()
    return model

def check_anatomical_consistency(boxes, labels, scores, threshold=0.5):
    """
    Check if predicted labels (L1-L5) descend correctly in the Y-axis.
    labels: [20, 21, 22, 23, 24] mapped to [1, 2, 3, 4, 5]
    """
    valid_dets = [(boxes[i][1], labels[i]) for i in range(len(boxes)) if scores[i] > threshold]
    if len(valid_dets) == 0:
        return False
        
    valid_dets.sort(key=lambda x: x[0]) # sort by Y-coordinate
    extracted_labels = [v[1] for v in valid_dets]
    
    # Check for duplicates
    if len(extracted_labels) != len(set(extracted_labels)):
        return False
        
    # Check strictly increasing (L1 above L2, etc)
    if extracted_labels != sorted(extracted_labels):
        return False
        
    return True

def process_vindr_transfer(device):
    vindr_manifest = pd.read_csv(r"c:\projects\spine\data\manifests\train.csv")
    out_dir_source = r"c:\projects\spine\outputs\phase12_domain_adaptation\source_only"
    out_dir_adapted = r"c:\projects\spine\outputs\phase12_domain_adaptation\adapted"
    os.makedirs(out_dir_source, exist_ok=True)
    os.makedirs(out_dir_adapted, exist_ok=True)
    
    source_model = get_source_only_model().to(device)
    dann_model = get_dann_model().to(device)
    
    vindr_ids = vindr_manifest['image_id'].unique()[:5] # Small qualitative set
    
    summary = []
    
    for img_id in vindr_ids:
        dicom_path = fr"C:\Users\Mohamed Riyas\.cache\kagglehub\datasets\siddhale937e92739\annotated-medical-image-dataset-for-spinal-lesions\versions\1\physionet.org\files\vindr-spinexr\1.0.0\train_images\{img_id}.dicom"
        if not os.path.exists(dicom_path): continue
        
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(float)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255
        pixel_array = pixel_array.astype(np.uint8)
        img = Image.fromarray(pixel_array).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_source = source_model(img_tensor)[0]
            pred_adapted = dann_model.faster_rcnn(img_tensor)[0]
            
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].imshow(img)
        axs[1].imshow(img)
        
        # Source Predictions
        boxes_s = pred_source['boxes'].cpu().numpy()
        scores_s = pred_source['scores'].cpu().numpy()
        labels_s = pred_source['labels'].cpu().numpy()
        s_count = 0
        s_conf = 0
        for i, box in enumerate(boxes_s):
            if scores_s[i] > 0.5:
                s_count += 1
                s_conf += scores_s[i]
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='cyan', facecolor='none')
                axs[0].add_patch(rect)
                axs[0].text(x1, y1, f"L{labels_s[i]} ({scores_s[i]:.2f})", color='cyan', fontsize=12)
        s_mean_conf = s_conf / max(s_count, 1)
        
        # Adapted Predictions
        boxes_a = pred_adapted['boxes'].cpu().numpy()
        scores_a = pred_adapted['scores'].cpu().numpy()
        labels_a = pred_adapted['labels'].cpu().numpy()
        a_count = 0
        a_conf = 0
        for i, box in enumerate(boxes_a):
            if scores_a[i] > 0.5:
                a_count += 1
                a_conf += scores_a[i]
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
                axs[1].add_patch(rect)
                axs[1].text(x1, y1, f"L{labels_a[i]} ({scores_a[i]:.2f})", color='lime', fontsize=12)
        a_mean_conf = a_conf / max(a_count, 1)
        
        axs[0].set_title("Source-Only (UNVALIDATED PREDICTION)")
        axs[1].set_title("DANN-Adapted (UNVALIDATED PREDICTION)")
        
        plt.savefig(os.path.join(out_dir_adapted, f"{img_id}_comparison.png"))
        plt.close()
        
        source_valid = check_anatomical_consistency(boxes_s, labels_s, scores_s)
        adapted_valid = check_anatomical_consistency(boxes_a, labels_a, scores_a)
        
        summary.append({
            "image_id": img_id,
            "source_num_detections": s_count,
            "adapted_num_detections": a_count,
            "source_ordering_valid": source_valid,
            "adapted_ordering_valid": adapted_valid,
            "source_mean_confidence": s_mean_conf,
            "adapted_mean_confidence": a_mean_conf,
            "qualitative_notes": "Awaiting manual review"
        })
        
    out_csv = r"c:\projects\spine\outputs\phase12_domain_adaptation\vindr_qualitative_summary.csv"
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f"VinDr Transfer evaluation saved to {out_csv}")

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    process_vindr_transfer(device)
