import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, r"c:\projects\spine")

from src.dataset.spine_dataset import SpineDataset
from src.dataset.transforms import BasicTransform
from src.models.cnn_baseline import CNNBaseline
from src.models.v3_nomorph import V3NoMorph
from src.models.neuromorphnet_v3 import NeuromorphnetV3

def calculate_pos_weight(dataset):
    targets = []
    no_finding_idx = dataset.no_finding_idx
    num_classes = dataset.num_classes
    
    for img_id in dataset.image_ids:
        group = dataset.image_groups.get_group(img_id)
        target_vec = np.zeros(num_classes, dtype=np.float32)
        has_pathology = False
        
        for _, row in group.iterrows():
            lbl_id = int(row['label_id'])
            target_vec[lbl_id] = 1.0
            if lbl_id != no_finding_idx:
                has_pathology = True
                
        if no_finding_idx != -1:
            if has_pathology:
                target_vec[no_finding_idx] = 0.0
            elif target_vec.sum() == 0:
                target_vec[no_finding_idx] = 1.0
                
        targets.append(target_vec)
    
    targets = np.array(targets)
    num_samples = len(targets)
    pos_counts = targets.sum(axis=0)
    neg_counts = num_samples - pos_counts
    
    pos_counts[pos_counts == 0] = 1
    pos_weight = neg_counts / pos_counts
    return torch.tensor(pos_weight, dtype=torch.float32)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def train_model(model_name, model, train_loader, val_loader, pos_weight, seed, epochs=10, lr=1e-4, out_dir=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    pos_weight = pos_weight.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    
    model_save_path = os.path.join(out_dir, "checkpoints", f"{model_name}_seed{seed}.pt")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"[{model_name} Seed {seed}] Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"[{model_name} Seed {seed}] Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f} Val: {val_loss:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            
    return history, model_save_path

def evaluate_model(model, model_save_path, test_loader, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            all_targets.append(targets.numpy())
            all_preds.append(preds.cpu().numpy())
            
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    
    per_class = {}
    for i, class_name in enumerate(class_names):
        tn, fp, fn, tp = confusion_matrix(all_targets[:, i], all_preds[:, i], labels=[0, 1]).ravel()
        per_class[class_name] = {
            "F1": report[class_name]["f1-score"],
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        }
        
    return {
        "Micro F1": report['micro avg']['f1-score'],
        "Macro F1": report['macro avg']['f1-score'],
        "Weighted F1": report['weighted avg']['f1-score'],
        "per_class": per_class
    }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_ablation():
    out_dir = r"c:\projects\spine\outputs\phase9_ablation"
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    
    with open(r"c:\projects\spine\data\class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    class_names = [v for k, v in sorted(class_mapping.items(), key=lambda item: int(item[0]))]
    
    tf = BasicTransform(size=(224, 224))
    train_ds = SpineDataset(r"c:\projects\spine\data\manifests\train.csv", 
                            r"c:\projects\spine\data\class_mapping.json", transforms=tf, return_morphology=False, return_graph=False)
    val_ds = SpineDataset(r"c:\projects\spine\data\manifests\val.csv", 
                          r"c:\projects\spine\data\class_mapping.json", transforms=tf, return_morphology=False, return_graph=False)
    test_ds = SpineDataset(r"c:\projects\spine\data\manifests\test.csv", 
                           r"c:\projects\spine\data\class_mapping.json", transforms=tf, return_morphology=False, return_graph=False)
                           
    invalid_csv = r"c:\projects\spine\docs\invalid_dicom_files.csv"
    if os.path.exists(invalid_csv):
        invalid_df = pd.read_csv(invalid_csv)
        invalid_ids = set(invalid_df['image_id'])
        test_ds.image_ids = [iid for iid in test_ds.image_ids if iid not in invalid_ids]
        
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    pos_weight = calculate_pos_weight(train_ds)
    
    baseline = CNNBaseline(num_classes=8)
    v3_nomorph = V3NoMorph(num_classes=8)
    v3_full = NeuromorphnetV3(num_classes=8)
    
    baseline_params = count_parameters(baseline)
    v3_nomorph_params = count_parameters(v3_nomorph)
    v3_full_params = count_parameters(v3_full)
    
    print(f"Parameters -> Baseline: {baseline_params}, V3-NoMorph: {v3_nomorph_params}, V3 Full: {v3_full_params}")
    
    # Evaluate existing baseline for reference (optional but good for report)
    baseline.load_state_dict(torch.load(r"c:\projects\spine\outputs\baseline\best_model.pt", map_location='cpu'))
    baseline_metrics = evaluate_model(baseline, r"c:\projects\spine\outputs\baseline\best_model.pt", test_loader, class_names)
    
    seeds = [42, 43, 44]
    results = []
    
    for seed in seeds:
        print(f"\n{'='*50}\nStarting Run for Seed {seed}\n{'='*50}")
        set_seed(seed)
        
        # Train V3-NoMorph
        model_nm = V3NoMorph(num_classes=8)
        hist_nm, path_nm = train_model("v3_nomorph", model_nm, train_loader, val_loader, pos_weight, seed, out_dir=out_dir)
        metrics_nm = evaluate_model(V3NoMorph(num_classes=8), path_nm, test_loader, class_names)
        
        results.append({
            "Model": "V3-NoMorph", "Seed": seed,
            "Micro F1": metrics_nm["Micro F1"], "Macro F1": metrics_nm["Macro F1"], "Weighted F1": metrics_nm["Weighted F1"],
            "per_class": metrics_nm["per_class"]
        })
        
        # Train V3 Full
        model_f = NeuromorphnetV3(num_classes=8)
        hist_f, path_f = train_model("v3_full", model_f, train_loader, val_loader, pos_weight, seed, out_dir=out_dir)
        metrics_f = evaluate_model(NeuromorphnetV3(num_classes=8), path_f, test_loader, class_names)
        
        results.append({
            "Model": "V3 Full", "Seed": seed,
            "Micro F1": metrics_f["Micro F1"], "Macro F1": metrics_f["Macro F1"], "Weighted F1": metrics_f["Weighted F1"],
            "per_class": metrics_f["per_class"]
        })
        
        # Save training curves for seed 42
        if seed == 42:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(hist_nm["train_loss"], label='V3-NoMorph Train')
            plt.plot(hist_nm["val_loss"], label='V3-NoMorph Val')
            plt.title('V3-NoMorph (Seed 42)')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(hist_f["train_loss"], label='V3 Full Train')
            plt.plot(hist_f["val_loss"], label='V3 Full Val')
            plt.title('V3 Full (Seed 42)')
            plt.legend()
            
            plt.savefig(os.path.join(out_dir, "training_curves.png"))
            plt.close()

    # Aggregate Results
    df_results = pd.DataFrame(results).drop(columns=['per_class'])
    df_results.to_csv(os.path.join(out_dir, "repeatability_results.csv"), index=False)
    
    summary = df_results.groupby('Model').agg({
        'Micro F1': ['mean', 'std'],
        'Macro F1': ['mean', 'std'],
        'Weighted F1': ['mean', 'std']
    }).reset_index()
    summary.columns = ['Model', 'Micro F1 Mean', 'Micro F1 Std', 'Macro F1 Mean', 'Macro F1 Std', 'Weighted F1 Mean', 'Weighted F1 Std']
    summary.to_csv(os.path.join(out_dir, "repeatability_summary.csv"), index=False)
    
    # Extract seed 42 for controlled comparison tables
    nm_42 = next(r for r in results if r["Model"] == "V3-NoMorph" and r["Seed"] == 42)
    f_42 = next(r for r in results if r["Model"] == "V3 Full" and r["Seed"] == 42)
    
    ablation_df = pd.DataFrame([
        {"Model": "ResNet18 Baseline", "Input": "Image", "Morphology Branch": "No", "Params": baseline_params, 
         "Micro F1": baseline_metrics["Micro F1"], "Macro F1": baseline_metrics["Macro F1"], "Weighted F1": baseline_metrics["Weighted F1"]},
        {"Model": "V3-NoMorph", "Input": "Image", "Morphology Branch": "No", "Params": v3_nomorph_params, 
         "Micro F1": nm_42["Micro F1"], "Macro F1": nm_42["Macro F1"], "Weighted F1": nm_42["Weighted F1"]},
        {"Model": "V3 Full", "Input": "Image", "Morphology Branch": "Yes", "Params": v3_full_params, 
         "Micro F1": f_42["Micro F1"], "Macro F1": f_42["Macro F1"], "Weighted F1": f_42["Weighted F1"]},
    ])
    ablation_df.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)
    
    # Per-class ablation for seed 42
    pc_data = []
    for class_name in class_names:
        b_f1 = baseline_metrics["per_class"][class_name]["F1"]
        nm_f1 = nm_42["per_class"][class_name]["F1"]
        f_f1 = f_42["per_class"][class_name]["F1"]
        
        nm_fp = nm_42["per_class"][class_name]["FP"]
        f_fp = f_42["per_class"][class_name]["FP"]
        nm_fn = nm_42["per_class"][class_name]["FN"]
        f_fn = f_42["per_class"][class_name]["FN"]
        
        pc_data.append({
            "Pathology": class_name,
            "Baseline F1": b_f1,
            "V3-NoMorph F1": nm_f1,
            "V3 F1": f_f1,
            "V3-NoMorph FP": nm_fp,
            "V3 FP": f_fp,
            "V3-NoMorph FN": nm_fn,
            "V3 FN": f_fn
        })
    pd.DataFrame(pc_data).to_csv(os.path.join(out_dir, "per_class_ablation.csv"), index=False)
    
    print("Ablation study complete.")

if __name__ == '__main__':
    run_ablation()
