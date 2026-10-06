import os
import torch
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_v2_model():
    num_classes = 6
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(r"c:\projects\spine\outputs\phase11_domain_adaptation\detector_v2.pt"))
    model.eval()
    return model

def evaluate_on_synthetic_v2(model, device):
    test_dir = r"c:\projects\spine\data\verse_drr_v2\test"
    imgs = os.listdir(test_dir)
    results = []
    
    out_dir = r"c:\projects\spine\outputs\phase11_domain_adaptation\v2_drr_visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    with torch.no_grad():
        for img_name in imgs:
            img_path = os.path.join(test_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
            
            prediction = model(img_tensor)[0]
            
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(img)
            
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            
            for i, box in enumerate(boxes):
                if scores[i] > 0.5:
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1, f"L{labels[i]} ({scores[i]:.2f})", color='green', fontsize=10)
                    
                    results.append({
                        "model": "V2_Advanced",
                        "image_id": img_name,
                        "pred_label": labels[i],
                        "score": scores[i],
                        "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2
                    })
            
            plt.title(f"V2 DRR Inference: {img_name}")
            plt.savefig(os.path.join(out_dir, f"{img_name}_inference.png"))
            plt.close()
            
    pd.DataFrame(results).to_csv(r"c:\projects\spine\outputs\phase11_domain_adaptation\drr_v1_vs_v2.csv", index=False)
    print("Evaluated V2 DRR tests.")

def process_vindr_transfer_v2(model, device):
    vindr_manifest = pd.read_csv(r"c:\projects\spine\data\manifests\train.csv")
    
    out_dir = r"c:\projects\spine\outputs\phase11_domain_adaptation\vindr_transfer"
    os.makedirs(out_dir, exist_ok=True)
    
    vindr_ids = vindr_manifest['image_id'].unique()[:3]
    
    for vindr_img_id in vindr_ids:
        dicom_path = fr"C:\Users\Mohamed Riyas\.cache\kagglehub\datasets\siddhale937e92739\annotated-medical-image-dataset-for-spinal-lesions\versions\1\physionet.org\files\vindr-spinexr\1.0.0\train_images\{vindr_img_id}.dicom"
        
        if not os.path.exists(dicom_path): continue
        
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(float)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255
        pixel_array = pixel_array.astype(np.uint8)
        
        img = Image.fromarray(pixel_array).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(img_tensor)[0]
            
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        # Plot V2 predictions
        for i, box in enumerate(boxes):
            if scores[i] > 0.5:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='cyan', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f"L{labels[i]} ({scores[i]:.2f})", color='cyan', fontsize=12)
                
        plt.title("V2 UNVALIDATED CROSS-DOMAIN INFERENCE")
        plt.savefig(os.path.join(out_dir, f"{vindr_img_id}_inference_v2.png"))
        plt.close()
        
    print("Evaluated V2 VinDr inference transfer.")

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_v2_model()
    model.to(device)
    
    evaluate_on_synthetic_v2(model, device)
    process_vindr_transfer_v2(model, device)
