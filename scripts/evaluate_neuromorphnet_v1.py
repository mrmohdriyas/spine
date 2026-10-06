import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, r"c:\projects\spine")

from src.dataset.spine_dataset import SpineDataset
from src.dataset.transforms import BasicTransform
from src.models.neuromorphnet_v1 import NeuromorphnetV1

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tf = BasicTransform(size=(224, 224))
    
    test_ds = SpineDataset(r"c:\projects\spine\data\manifests\test.csv", 
                           r"c:\projects\spine\data\class_mapping.json", 
                           transforms=tf, return_morphology=True)
                           
    invalid_csv = r"c:\projects\spine\docs\invalid_dicom_files.csv"
    if os.path.exists(invalid_csv):
        invalid_df = pd.read_csv(invalid_csv)
        if not invalid_df.empty:
            invalid_ids = set(invalid_df['image_id'])
            # Filter them out of test_ds
            test_ds.image_ids = [iid for iid in test_ds.image_ids if iid not in invalid_ids]
                           
    print(f"Total valid test images for evaluation: {len(test_ds.image_ids)}")
    
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    model = NeuromorphnetV1(num_classes=test_ds.num_classes).to(device)
    
    model_path = r"c:\projects\spine\outputs\neuromorphnet_v1\best_model.pt"
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets, morphs in test_loader:
            images, targets, morphs = images.to(device), targets.to(device), morphs.to(device)
            logits = model(images, morphs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    
    # Class names
    with open(r"c:\projects\spine\data\class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    
    # Sort class names by index
    class_names = [v for k, v in sorted(class_mapping.items(), key=lambda item: int(item[0]))]
    
    # Compute metrics
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    
    auroc_scores = {}
    for i, class_name in enumerate(class_names):
        try:
            # AUROC is only valid if both classes are present in true labels
            auroc = roc_auc_score(all_targets[:, i], all_probs[:, i])
        except ValueError:
            auroc = np.nan
        auroc_scores[class_name] = auroc
        
    # Print high-level metrics
    print("Evaluation completed successfully.")
    print(f"Micro F1: {report['micro avg']['f1-score']:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    
    # Save full report
    output_dir = r"c:\projects\spine\outputs\neuromorphnet_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Add AUROC to report
    for class_name in class_names:
        report[class_name]['auroc'] = auroc_scores[class_name]
        
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_dir, "classification_report.csv"))
    
    # Generate per-class confusion matrix
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        cm = confusion_matrix(all_targets[:, i], all_preds[:, i])
        if cm.shape == (2, 2):
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues', 
                        xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
            axes[i].set_title(class_name)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        else:
            axes[i].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            axes[i].set_title(class_name)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    metrics = {
        "micro_f1": report['micro avg']['f1-score'],
        "macro_f1": report['macro avg']['f1-score'],
        "weighted_f1": report['weighted avg']['f1-score']
    }
    
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
if __name__ == '__main__':
    evaluate()
