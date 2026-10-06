import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import sys
sys.path.insert(0, r"c:\projects\spine")

from src.dataset.spine_dataset import SpineDataset
from src.dataset.transforms import BasicTransform
from src.models.neuromorphnet_v2 import NeuromorphnetV2

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    morphs = torch.stack([item[2] for item in batch])
    graphs = [item[3] for item in batch]
    return images, targets, morphs, graphs

def visualize_graphs(dataset, model, device, output_dir, num_samples=10):
    vis_dir = os.path.join(output_dir, "graph_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Pick first few images with multiple annotations
    visualized = 0
    for idx in range(len(dataset)):
        if visualized >= num_samples:
            break
            
        image, target, morph, graph = dataset[idx]
        nodes = graph['nodes']
        if len(nodes) < 2:
            continue
            
        # Draw on original image
        image_id = dataset.image_ids[idx]
        group = dataset.image_groups.get_group(image_id)
        image_path = group.iloc[0]['image_path']
        
        orig_img, orig_h, orig_w = dataset._load_dicom(image_path)
        
        # Convert to BGR 8-bit for cv2 drawing
        img_draw = (orig_img * 255).astype(np.uint8)
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)
        
        # Draw nodes (boxes)
        centers = []
        for i, row in group.iterrows():
            if not pd.isna(row.get('x_min')):
                xmin = int(float(row['x_min']))
                ymin = int(float(row['y_min']))
                xmax = int(float(row['x_max']))
                ymax = int(float(row['y_max']))
                cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2
                centers.append((cx, cy))
                cv2.circle(img_draw, (cx, cy), 5, (0, 0, 255), -1)
                
        # Draw edges
        edges = graph['edge_indices']
        if edges.shape[1] > 0:
            for e in range(edges.shape[1]):
                u = int(edges[0, e])
                v = int(edges[1, e])
                pt1 = centers[u]
                pt2 = centers[v]
                cv2.line(img_draw, pt1, pt2, (255, 0, 0), 2)
                
        cv2.imwrite(os.path.join(vis_dir, f"{image_id}.png"), img_draw)
        visualized += 1

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tf = BasicTransform(size=(224, 224))
    
    test_ds = SpineDataset(r"c:\projects\spine\data\manifests\test.csv", 
                           r"c:\projects\spine\data\class_mapping.json", 
                           transforms=tf, return_morphology=True, return_graph=True)
                           
    invalid_csv = r"c:\projects\spine\docs\invalid_dicom_files.csv"
    if os.path.exists(invalid_csv):
        invalid_df = pd.read_csv(invalid_csv)
        if not invalid_df.empty:
            invalid_ids = set(invalid_df['image_id'])
            test_ds.image_ids = [iid for iid in test_ds.image_ids if iid not in invalid_ids]
                           
    print(f"Total valid test images for evaluation: {len(test_ds.image_ids)}")
    
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    model = NeuromorphnetV2(num_classes=test_ds.num_classes).to(device)
    
    output_dir = r"c:\projects\spine\outputs\neuromorphnet_v2"
    model_path = os.path.join(output_dir, "best_model.pt")
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets, morphs, graphs in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            morphs = morphs.to(device)
            
            for g in graphs:
                g['nodes'] = g['nodes'].to(device)
                g['edge_indices'] = g['edge_indices'].to(device)
                g['edge_features'] = g['edge_features'].to(device)
                
            logits = model(images, morphs, graphs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    
    with open(r"c:\projects\spine\data\class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    
    class_names = [v for k, v in sorted(class_mapping.items(), key=lambda item: int(item[0]))]
    
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    
    auroc_scores = {}
    for i, class_name in enumerate(class_names):
        try:
            auroc = roc_auc_score(all_targets[:, i], all_probs[:, i])
        except ValueError:
            auroc = np.nan
        auroc_scores[class_name] = auroc
        
    print("Evaluation completed successfully.")
    print(f"Micro F1: {report['micro avg']['f1-score']:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in class_names:
        report[class_name]['auroc'] = auroc_scores[class_name]
        
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_dir, "classification_report.csv"))
    
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
        
    print("Generating graph visualizations...")
    visualize_graphs(test_ds, model, device, output_dir, num_samples=10)
        
if __name__ == '__main__':
    evaluate()
