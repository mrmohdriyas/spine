import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from src.data.dataset import VinDrDataset, collate_fn
from src.models.neuromorphnet_final import AnatomyConstrainedDualGraphNeuroMorphNet

def evaluate_uncertainty():
    print("Part G: Uncertainty Evaluation (UGDM)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Dataset
    full_val = VinDrDataset(r"c:\projects\spine\data\manifests\test.csv", r"c:\projects\spine\data\vindr-spinexr\test_images", augment=False)
    test_subset = Subset(full_val, list(range(50)))
    test_loader = DataLoader(test_subset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # 2. Load Final Model
    model = AnatomyConstrainedDualGraphNeuroMorphNet(embed_dim=256, num_pathologies=8).to(device)
    chkpt_path = r"c:\projects\spine\outputs\phase15_validation\checkpoints\best_pathology_model.pt"
    if os.path.exists(chkpt_path):
        model.load_state_dict(torch.load(chkpt_path, map_location=device))
    model.eval()
    
    all_targets = []
    all_preds = []
    all_uncertainty = []
    all_confidence = []
    
    print("Running Monte Carlo Dropout Inference...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets_tensor = torch.stack([t['labels'] for t in targets])
            
            # Use UGDM
            out = model(images, return_uncertainty=True)
            
            preds = (out['mean_prob'] > 0.5).int().cpu()
            
            all_preds.append(preds)
            all_targets.append(targets_tensor)
            all_uncertainty.append(out['uncertainty'].cpu())
            all_confidence.append(out['confidence'].cpu())
            
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_uncertainty = torch.cat(all_uncertainty, dim=0).numpy()
    all_confidence = torch.cat(all_confidence, dim=0).numpy()
    
    classes = [
        "Osteophytes", "No finding", "Disc space narrowing", "Other lesions",
        "Surgical implant", "Foraminal stenosis", "Spondylolysthesis", "Vertebral collapse"
    ]
    
    results = []
    
    # Analyze by class
    for i, cls_name in enumerate(classes):
        cls_targets = all_targets[:, i]
        cls_preds = all_preds[:, i]
        cls_unc = all_uncertainty[:, i]
        cls_conf = all_confidence[:, i]
        
        correct_mask = (cls_targets == cls_preds)
        incorrect_mask = (cls_targets != cls_preds)
        
        mean_unc_correct = cls_unc[correct_mask].mean() if correct_mask.any() else 0.0
        mean_unc_incorrect = cls_unc[incorrect_mask].mean() if incorrect_mask.any() else 0.0
        
        mean_conf_correct = cls_conf[correct_mask].mean() if correct_mask.any() else 0.0
        mean_conf_incorrect = cls_conf[incorrect_mask].mean() if incorrect_mask.any() else 0.0
        
        results.append({
            "Class": cls_name,
            "Mean_Uncertainty_Overall": cls_unc.mean(),
            "Mean_Confidence_Overall": cls_conf.mean(),
            "Mean_Uncertainty_Correct": mean_unc_correct,
            "Mean_Uncertainty_Incorrect": mean_unc_incorrect,
            "Higher_Uncertainty_On_Errors": mean_unc_incorrect > mean_unc_correct
        })
        
    out_dir = r"c:\projects\spine\outputs\phase15_validation"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(out_dir, "uncertainty.csv"), index=False)
    
    print("Uncertainty evaluation complete. Results saved to outputs/phase15_validation/uncertainty.csv")

if __name__ == "__main__":
    evaluate_uncertainty()
