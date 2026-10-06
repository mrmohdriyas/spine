import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader, Subset
from src.data.dataset import VinDrDataset, collate_fn
from src.models.neuromorphnet_final import AnatomyConstrainedDualGraphNeuroMorphNet

def evaluate():
    print("Starting Final NeuroMorphNet Evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = r"c:\projects\spine\outputs\final_neuromorphnet"
    
    # 1. Load Data
    full_test = VinDrDataset(r"c:\projects\spine\data\manifests\test.csv", r"c:\projects\spine\data\vindr-spinexr\test_images", augment=False)
    subset_size = min(50, len(full_test))
    test_subset = Subset(full_test, list(range(subset_size)))
    test_loader = DataLoader(test_subset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # 2. Load Model
    model = AnatomyConstrainedDualGraphNeuroMorphNet(embed_dim=256, num_pathologies=8).to(device)
    chkpt_path = os.path.join(out_dir, "best_model.pt")
    if os.path.exists(chkpt_path):
        model.load_state_dict(torch.load(chkpt_path, map_location=device))
    else:
        print(f"Warning: {chkpt_path} not found. Running with untrained weights.")
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    all_unc = []
    
    from tqdm import tqdm
    print("Running Inference...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            targets_tensor = torch.stack([t['labels'] for t in targets])
            
            # Using UGDM for uncertainty evaluation
            out = model(images, return_uncertainty=True)
            
            probs = out['mean_prob'].cpu()
            preds = (probs > 0.5).int()
            
            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(targets_tensor)
            all_unc.append(out['uncertainty'].cpu())
            
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_unc = torch.cat(all_unc, dim=0).numpy()
    
    classes = [
        "Osteophytes", "No finding", "Disc space narrowing", "Other lesions",
        "Surgical implant", "Foraminal stenosis", "Spondylolysthesis", "Vertebral collapse"
    ]
    
    # 3. Overall Metrics
    micro_f1 = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    metrics_json = {
        "Micro_F1": float(micro_f1),
        "Macro_F1": float(macro_f1),
        "Weighted_F1": float(weighted_f1)
    }
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=4)
        
    print(f"Overall Metrics: Micro={micro_f1:.4f}, Macro={macro_f1:.4f}, Weighted={weighted_f1:.4f}")
    
    # 4. Per-Class Metrics
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    
    per_class_list = []
    for i, cls in enumerate(classes):
        per_class_list.append({
            "Class": cls,
            "Precision": precision[i],
            "Recall": recall[i],
            "F1": f1[i],
            "Support": support[i] if support is not None else 0
        })
    pd.DataFrame(per_class_list).to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    
    # 5. Final Comparison
    comparison = [
        {"Model": "ResNet18 baseline", "Micro_F1": 0.4000, "Macro_F1": 0.2751, "Weighted_F1": 0.5970},
        {"Model": "NeuroMorphNet V3", "Micro_F1": 0.4906, "Macro_F1": 0.3614, "Weighted_F1": 0.6473},
        {"Model": "Final NeuroMorphNet", "Micro_F1": micro_f1, "Macro_F1": macro_f1, "Weighted_F1": weighted_f1}
    ]
    pd.DataFrame(comparison).to_csv(os.path.join(out_dir, "final_comparison.csv"), index=False)
    
    # 6. Per-Class Comparison (vs Baseline and V3 theoretical limits, here we just document the final per user request structure,
    # or pad missing numbers with 'unknown' for prior models since they aren't provided in the prompt for per-class)
    per_class_comp = []
    for i, cls in enumerate(classes):
        per_class_comp.append({
            "Class": cls,
            "Baseline_F1": "N/A", 
            "V3_F1": "N/A", 
            "Final_Precision": precision[i],
            "Final_Recall": recall[i],
            "Final_F1": f1[i]
        })
    pd.DataFrame(per_class_comp).to_csv(os.path.join(out_dir, "per_class_comparison.csv"), index=False)
    
    # 7. Error Analysis Cases
    errors = []
    for i in range(len(all_targets)):
        for j, cls in enumerate(classes):
            target = all_targets[i, j]
            pred = all_preds[i, j]
            if target == 1 and pred == 1:
                type_ = "TP"
            elif target == 0 and pred == 1:
                type_ = "FP"
            elif target == 1 and pred == 0:
                type_ = "FN"
            else:
                continue
            
            errors.append({
                "Image_Idx": i,
                "Class": cls,
                "Type": type_,
                "Probability": float(all_probs[i, j])
            })
            if len(errors) > 50:
                break
        if len(errors) > 50:
            break
            
    pd.DataFrame(errors).to_csv(os.path.join(out_dir, "error_cases.csv"), index=False)
    
    # 8. Uncertainty
    unc_results = []
    for i, cls in enumerate(classes):
        cls_probs = all_probs[:, i]
        cls_unc = all_unc[:, i]
        unc_results.append({
            "Class": cls,
            "Mean_Probability": cls_probs.mean(),
            "Variance": cls_unc.var(),
            "Uncertainty": cls_unc.mean()
        })
    pd.DataFrame(unc_results).to_csv(os.path.join(out_dir, "uncertainty.csv"), index=False)
    
    # 9. Extract Visualization Data
    vis_dir = os.path.join(out_dir, "visualization_data")
    os.makedirs(vis_dir, exist_ok=True)
    
    pred_data = []
    gt_data = []
    
    for i in range(len(all_targets)):
        pred_row = {"image_id": f"test_img_{i}"}
        gt_row = {"image_id": f"test_img_{i}"}
        for j, cls in enumerate(classes):
            safe_cls = cls.lower().replace(" ", "_")
            pred_row[f"prob_{safe_cls}"] = float(all_probs[i, j])
            pred_row[f"pred_{safe_cls}"] = int(all_preds[i, j])
            gt_row[f"true_{safe_cls}"] = int(all_targets[i, j])
            
        pred_data.append(pred_row)
        gt_data.append(gt_row)
        
    pd.DataFrame(pred_data).to_csv(os.path.join(vis_dir, "predictions.csv"), index=False)
    pd.DataFrame(gt_data).to_csv(os.path.join(vis_dir, "ground_truth.csv"), index=False)
    
    print("Evaluation Complete. All artifacts saved.")

if __name__ == "__main__":
    evaluate()
