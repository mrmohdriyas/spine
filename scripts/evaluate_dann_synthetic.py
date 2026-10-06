import os
import torch
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

def evaluate_models(device):
    test_dir = r"c:\projects\spine\data\verse_drr_v2\test"
    imgs = os.listdir(test_dir)
    results = []
    
    source_model = get_source_only_model().to(device)
    dann_model = get_dann_model().to(device)
    
    with torch.no_grad():
        for img_name in imgs:
            img_path = os.path.join(test_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
            
            # Source Only
            pred_source = source_model(img_tensor)[0]
            boxes = pred_source['boxes'].cpu().numpy()
            scores = pred_source['scores'].cpu().numpy()
            labels = pred_source['labels'].cpu().numpy()
            for i, box in enumerate(boxes):
                if scores[i] > 0.5:
                    results.append({
                        "model": "Source-Only",
                        "image_id": img_name,
                        "pred_label": labels[i],
                        "score": scores[i],
                        "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3]
                    })
                    
            # DANN Adapted
            pred_dann = dann_model.faster_rcnn(img_tensor)[0]
            boxes = pred_dann['boxes'].cpu().numpy()
            scores = pred_dann['scores'].cpu().numpy()
            labels = pred_dann['labels'].cpu().numpy()
            for i, box in enumerate(boxes):
                if scores[i] > 0.5:
                    results.append({
                        "model": "DANN-Adapted",
                        "image_id": img_name,
                        "pred_label": labels[i],
                        "score": scores[i],
                        "x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3]
                    })
            
    out_csv = r"c:\projects\spine\outputs\phase12_domain_adaptation\source_vs_adapted.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Synthetic evaluation saved to {out_csv}")

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    evaluate_models(device)
