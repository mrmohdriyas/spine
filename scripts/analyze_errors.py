import os
import torch
import json
import pandas as pd
from PIL import Image
import torchvision
from evaluate_real_xray import get_dann_model, get_source_only_model
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def analyze_errors():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    out_dir = r"c:\projects\spine\outputs\phase13_real_xray_validation"
    
    # Load Models
    source_model = get_source_only_model().to(device)
    
    dann_ft = get_dann_model().faster_rcnn
    in_features = dann_ft.roi_heads.box_predictor.cls_score.in_features
    dann_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, 6)
    dann_ft.load_state_dict(torch.load(os.path.join(out_dir, "detector_dann_finetuned.pt"), weights_only=True))
    dann_ft.to(device)
    dann_ft.eval()
    
    models = {"Source-Only": source_model, "DANN-Finetuned": dann_ft}
    
    with open(r"c:\projects\spine\data\vindr_vertebral_annotations\test.json", 'r') as f:
        test_data = json.load(f)
        
    error_cases = []
    consistency_results = []
    
    for item in test_data:
        img_id = item['image_id']
        img_path = os.path.join(r"c:\projects\spine\data\vindr_vertebral_annotations\images", f"{img_id}.png")
        if not os.path.exists(img_path): continue
        
        img = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
        
        for model_name, model in models.items():
            with torch.no_grad():
                pred = model(img_tensor)[0]
                
            boxes_pred = pred['boxes'].cpu().numpy()
            scores_pred = pred['scores'].cpu().numpy()
            labels_pred = pred['labels'].cpu().numpy()
            
            # Filter by conf
            valid_idx = scores_pred > 0.5
            boxes = boxes_pred[valid_idx]
            scores = scores_pred[valid_idx]
            labels = labels_pred[valid_idx]
            
            # Error Case checking (simplified representation)
            if len(boxes) == 0 and len(item['annotations']) > 0:
                error_cases.append({
                    "model": model_name,
                    "image_id": img_id,
                    "error_type": "False Negative (No detections)",
                    "gt_count": len(item['annotations'])
                })
            
            # Anatomical Ordering
            dets = [(boxes[i][1], labels[i]) for i in range(len(boxes))]
            dets.sort(key=lambda x: x[0]) # sort by Y
            extracted_labels = [v[1] for v in dets]
            
            has_duplicates = len(extracted_labels) != len(set(extracted_labels))
            is_ordered = extracted_labels == sorted(extracted_labels)
            
            consistency_results.append({
                "model": model_name,
                "image_id": img_id,
                "num_detections": len(boxes),
                "has_duplicates": has_duplicates,
                "valid_ordering": is_ordered and not has_duplicates and len(boxes) > 0
            })
            
    pd.DataFrame(error_cases).to_csv(os.path.join(out_dir, "error_cases.csv"), index=False)
    pd.DataFrame(consistency_results).to_csv(os.path.join(out_dir, "anatomical_consistency.csv"), index=False)
    print("Error analysis complete.")

if __name__ == "__main__":
    analyze_errors()
