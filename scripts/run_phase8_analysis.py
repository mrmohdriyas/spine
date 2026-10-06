import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA

sys.path.insert(0, r"c:\projects\spine")
from src.dataset.spine_dataset import SpineDataset
from src.dataset.transforms import BasicTransform
from src.models.cnn_baseline import CNNBaseline
from src.models.neuromorphnet_v3 import NeuromorphnetV3

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        
        score = logits[0, class_idx]
        score.backward(retain_graph=True)
        
        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.squeeze().cpu().detach().numpy()

def generate_heatmap_overlay(img_path, cam, dataset):
    orig_img, orig_h, orig_w = dataset._load_dicom(img_path)
    img_draw = (orig_img * 255).astype(np.uint8)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)
    
    heatmap = cv2.resize(cam, (orig_w, orig_h))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed = cv2.addWeighted(img_draw, 0.6, heatmap, 0.4, 0)
    return superimposed

def run_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    out_dir = r"c:\projects\spine\outputs\neuromorphnet_v3\explainability"
    os.makedirs(os.path.join(out_dir, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "gradcam"), exist_ok=True)
    
    with open(r"c:\projects\spine\data\class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    class_names = [v for k, v in sorted(class_mapping.items(), key=lambda item: int(item[0]))]
    
    # Dataset
    tf = BasicTransform(size=(224, 224))
    test_ds = SpineDataset(r"c:\projects\spine\data\manifests\test.csv", 
                           r"c:\projects\spine\data\class_mapping.json", 
                           transforms=tf, return_morphology=False, return_graph=False)
                           
    invalid_csv = r"c:\projects\spine\docs\invalid_dicom_files.csv"
    if os.path.exists(invalid_csv):
        invalid_df = pd.read_csv(invalid_csv)
        invalid_ids = set(invalid_df['image_id'])
        test_ds.image_ids = [iid for iid in test_ds.image_ids if iid not in invalid_ids]
        
    print(f"Total valid test images: {len(test_ds.image_ids)}")
    
    # Models
    baseline = CNNBaseline(num_classes=8).to(device)
    baseline.load_state_dict(torch.load(r"c:\projects\spine\outputs\baseline\best_model.pt", map_location=device))
    baseline.eval()
    
    v3 = NeuromorphnetV3(num_classes=8).to(device)
    v3.load_state_dict(torch.load(r"c:\projects\spine\outputs\neuromorphnet_v3\best_model.pt", map_location=device))
    v3.eval()
    
    # Storage
    all_targets = []
    base_preds = []
    v3_preds = []
    v3_probs = []
    v3_morphs = []
    
    image_paths = []
    image_ids = []
    
    # Inference
    print("Running inference...")
    for idx in range(len(test_ds)):
        img, target = test_ds[idx]
        image_id = test_ds.image_ids[idx]
        img_path = test_ds.image_groups.get_group(image_id).iloc[0]['image_path']
        
        image_paths.append(img_path)
        image_ids.append(image_id)
        
        img_t = img.unsqueeze(0).to(device)
        target_t = target.unsqueeze(0).to(device)
        
        with torch.no_grad():
            b_logits = baseline(img_t)
            b_pred = (torch.sigmoid(b_logits) > 0.5).float()
            
            v3_logits = v3(img_t)
            v3_prob = torch.sigmoid(v3_logits)
            v3_pred = (v3_prob > 0.5).float()
            
            # Extract morphology embedding from V3
            feature_map = v3.backbone(img_t)
            morph_emb = v3.morphology_branch(feature_map)
            
        all_targets.append(target_t.cpu().numpy()[0])
        base_preds.append(b_pred.cpu().numpy()[0])
        v3_preds.append(v3_pred.cpu().numpy()[0])
        v3_probs.append(v3_prob.cpu().numpy()[0])
        v3_morphs.append(morph_emb.cpu().numpy()[0])
        
    all_targets = np.array(all_targets)
    base_preds = np.array(base_preds)
    v3_preds = np.array(v3_preds)
    v3_probs = np.array(v3_probs)
    v3_morphs = np.array(v3_morphs)
    
    # Metrics & Confusion Matrices
    print("Generating metrics and confusion matrices...")
    v3_report = classification_report(all_targets, v3_preds, target_names=class_names, output_dict=True, zero_division=0)
    base_report = classification_report(all_targets, base_preds, target_names=class_names, output_dict=True, zero_division=0)
    
    per_class_metrics = []
    baseline_vs_v3 = []
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        y_true = all_targets[:, i]
        y_pred = v3_preds[:, i]
        b_pred = base_preds[:, i]
        
        # V3 Stats
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        support = int(np.sum(y_true))
        
        try: auroc = roc_auc_score(y_true, v3_probs[:, i])
        except: auroc = np.nan
        
        per_class_metrics.append({
            "Pathology": class_name,
            "Precision": v3_report[class_name]["precision"],
            "Recall": v3_report[class_name]["recall"],
            "F1": v3_report[class_name]["f1-score"],
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Support": support,
            "AUROC": auroc
        })
        
        # Baseline vs V3
        b_tn, b_fp, b_fn, b_tp = confusion_matrix(y_true, b_pred, labels=[0, 1]).ravel()
        b_f1 = base_report[class_name]["f1-score"]
        v3_f1 = v3_report[class_name]["f1-score"]
        
        baseline_vs_v3.append({
            "Pathology": class_name,
            "Baseline F1": b_f1,
            "V3 F1": v3_f1,
            "Difference": v3_f1 - b_f1,
            "Baseline TP": b_tp, "V3 TP": tp,
            "Baseline FP": b_fp, "V3 FP": fp,
            "Baseline FN": b_fn, "V3 FN": fn
        })
        
        # Confusion matrix plot
        cm_matrix = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm_matrix, annot=True, fmt='d', ax=axes[i], cmap='Blues', 
                    xticklabels=['False', 'True'], yticklabels=['False', 'True'])
        axes[i].set_title(class_name)
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrices", "combined_matrices.png"))
    plt.close()
    
    pd.DataFrame(per_class_metrics).to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    pd.DataFrame(baseline_vs_v3).to_csv(os.path.join(out_dir, "baseline_vs_v3.csv"), index=False)
    
    # Error Cases Collection
    print("Collecting error cases...")
    error_cases = []
    
    # Pick classes that have annotations for Grad-CAM
    cam_classes = ["Osteophytes", "Surgical implant"]
    cam_targets = {c: {"TP": None, "FP": None, "FN": None} for c in cam_classes}
    
    for i, class_name in enumerate(class_names):
        y_true = all_targets[:, i]
        y_pred = v3_preds[:, i]
        
        for idx in range(len(y_true)):
            gt = y_true[idx]
            pred = y_pred[idx]
            prob = v3_probs[idx, i]
            
            error_type = "Correct Negative"
            if gt == 1 and pred == 1: error_type = "True Positive"
            elif gt == 0 and pred == 1: error_type = "False Positive"
            elif gt == 1 and pred == 0: error_type = "False Negative"
            
            # Save for Grad-CAM
            if class_name in cam_classes:
                if error_type == "True Positive" and cam_targets[class_name]["TP"] is None:
                    cam_targets[class_name]["TP"] = idx
                elif error_type == "False Positive" and cam_targets[class_name]["FP"] is None:
                    cam_targets[class_name]["FP"] = idx
                elif error_type == "False Negative" and cam_targets[class_name]["FN"] is None:
                    cam_targets[class_name]["FN"] = idx
                    
            if error_type != "Correct Negative": # Save interesting cases
                error_cases.append({
                    "Image ID": image_ids[idx],
                    "Pathology": class_name,
                    "Ground Truth": gt,
                    "Prediction": pred,
                    "Probability": prob,
                    "Error Type": error_type
                })
                
    pd.DataFrame(error_cases).to_csv(os.path.join(out_dir, "error_cases.csv"), index=False)
    
    # Grad-CAM
    print("Generating Grad-CAM...")
    # Hook into last layer of ResNet backbone (layer4[-1])
    target_layer = v3.backbone[-1][-1]
    gradcam = GradCAM(v3, target_layer)
    
    for class_name, targets in cam_targets.items():
        class_idx = class_names.index(class_name)
        for err_type, idx in targets.items():
            if idx is not None:
                img_path = image_paths[idx]
                img_t = test_ds[idx][0].unsqueeze(0).to(device)
                
                cam = gradcam(img_t, class_idx)
                overlay = generate_heatmap_overlay(img_path, cam, test_ds)
                
                out_name = f"{class_name.lower().replace(' ', '_')}_{err_type.lower()}_{image_ids[idx][:8]}.png"
                cv2.imwrite(os.path.join(out_dir, "gradcam", out_name), overlay)

    # PCA
    print("Generating PCA...")
    pca = PCA(n_components=2)
    morph_pca = pca.fit_transform(v3_morphs)
    
    plt.figure(figsize=(10, 8))
    # Color by Osteophytes (class 3) for example
    scatter = plt.scatter(morph_pca[:, 0], morph_pca[:, 1], c=all_targets[:, 3], cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='Osteophytes GT')
    plt.title('PCA of Image-Derived Morphology Embedding')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(os.path.join(out_dir, "morphology_embedding_pca.png"))
    plt.close()
    
    print("Analysis complete.")

if __name__ == '__main__':
    run_analysis()
