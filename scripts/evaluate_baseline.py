import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             roc_auc_score, classification_report, multilabel_confusion_matrix)

import sys
sys.path.insert(0, r"c:\projects\spine")

from src.dataset.spine_dataset import SpineDataset
from src.dataset.transforms import BasicTransform
from src.models.cnn_baseline import CNNBaseline

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tf = BasicTransform(size=(224, 224))
    
    test_ds = SpineDataset(r"c:\projects\spine\data\manifests\test.csv", 
                           r"c:\projects\spine\data\class_mapping.json", 
                           transforms=tf)
                           
    invalid_csv = r"c:\projects\spine\docs\invalid_dicom_files.csv"
    if os.path.exists(invalid_csv):
        invalid_df = pd.read_csv(invalid_csv)
        if not invalid_df.empty:
            invalid_ids = set(invalid_df['image_id'])
            # Filter them out of test_ds
            test_ds.image_ids = [iid for iid in test_ds.image_ids if iid not in invalid_ids]
                           
    print(f"Total valid test images for evaluation: {len(test_ds.image_ids)}")
    
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    model = CNNBaseline(num_classes=test_ds.num_classes).to(device)
    model.load_state_dict(torch.load(r"c:\projects\spine\outputs\baseline\best_model.pt", map_location=device, weights_only=True))
    model.eval()
    
    all_targets = []
    all_probs = []
    all_preds = []
    all_image_ids = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
            # Since Dataset gets groups by index, let's extract image_ids manually
            start_idx = i * 16
            end_idx = min(start_idx + len(images), len(test_ds))
            for j in range(start_idx, end_idx):
                all_image_ids.append(test_ds.image_ids[j])
            
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    
    # Calculate metrics
    micro_f1 = f1_score(all_targets, all_preds, average='micro')
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    class_names = [test_ds.class_mapping[str(i)] for i in range(test_ds.num_classes)]
    
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    
    # Calculate AUROC safely
    roc_aucs = {}
    for i, c in enumerate(class_names):
        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            roc_aucs[c] = auc
            report_dict[c]['roc_auc'] = auc
        except ValueError:
            roc_aucs[c] = None
            report_dict[c]['roc_auc'] = None
    
    metrics = {
        "Micro F1": micro_f1,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    }
    
    out_dir = r"c:\projects\spine\outputs\baseline"
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Classification report CSV
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(out_dir, "classification_report.csv"))
    
    # Multilabel confusion matrix
    mcm = multilabel_confusion_matrix(all_targets, all_preds)
    fig, axes = plt.subplots(int(np.ceil(test_ds.num_classes/4)), 4, figsize=(15, 6))
    axes = axes.ravel()
    for i, cm in enumerate(mcm):
        ax = axes[i]
        cax = ax.matshow(cm, cmap='Blues')
        for (y, x), val in np.ndenumerate(cm):
            ax.text(x, y, f"{val}", ha='center', va='center')
        ax.set_title(class_names[i])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Neg', 'Pos'])
        ax.set_yticklabels(['Neg', 'Pos'])
    
    # Hide unused axes
    for j in range(test_ds.num_classes, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()
    
    # Predictions CSV
    preds_records = []
    for i in range(len(all_image_ids)):
        img_id = all_image_ids[i]
        true_labels = [class_names[c] for c in range(test_ds.num_classes) if all_targets[i, c] == 1]
        pred_labels = [class_names[c] for c in range(test_ds.num_classes) if all_preds[i, c] == 1]
        
        preds_records.append({
            "image_id": img_id,
            "true_labels": ", ".join(true_labels) if true_labels else "None",
            "predicted_labels": ", ".join(pred_labels) if pred_labels else "None",
            "confidences": ", ".join([f"{class_names[c]}:{all_probs[i, c]:.2f}" for c in range(test_ds.num_classes)])
        })
        
    preds_df = pd.DataFrame(preds_records)
    preds_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    
    # Training curves
    with open(os.path.join(out_dir, "training_history.json"), "r") as f:
        history = json.load(f)
        
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCEWithLogits)')
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()
    
    print("Evaluation completed successfully.")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

if __name__ == "__main__":
    evaluate()
