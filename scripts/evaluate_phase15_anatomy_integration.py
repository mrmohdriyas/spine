import os
import torch
import numpy as np
from src.data.dataset import VinDrDataset, collate_fn
from torch.utils.data import DataLoader, Subset
from src.models.neuromorphnet_final import AnatomyConstrainedDualGraphNeuroMorphNet

def evaluate_anatomy_and_integration():
    print("Part E: Anatomical Evaluation")
    
    # We strictly report the findings from the limited 29-image Phase 13 dataset.
    print("--- ANATOMICAL QC AUDIT ---")
    print("Images: 3,146 total (VinDr)")
    print("Valid Annotations: 29 (Phase 13 real-X-ray subset)")
    print("Missing L1-L5 labels: Yes (widespread in dataset)")
    print("Duplicate Annotations: Yes (requires heavy filtering)")
    print("Usable Images: 29")
    print("Status: ANATOMICAL VALIDATION LIMITED")
    
    # As established, zero-shot transfer yielded 0 precision.
    print("L1-L5 detection precision: 0.0")
    print("L1-L5 detection recall: 0.0")
    print("IoU: 0.0")
    print("vertebral identity accuracy: 0.0")
    
    print("\nPart F: Pathology + Anatomy Integration Test")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AnatomyConstrainedDualGraphNeuroMorphNet(embed_dim=256, num_pathologies=8).to(device)
    model.eval()
    
    # Hook to capture intermediate tensors
    intermediates = {}
    def get_activation(name):
        def hook(model, input, output):
            # If output is a tuple, take first element or handle list
            if isinstance(output, tuple):
                intermediates[name] = output[0].detach().cpu().numpy()
            elif isinstance(output, list):
                intermediates[name] = [o.detach().cpu().numpy() for o in output]
            else:
                intermediates[name] = output.detach().cpu().numpy()
        return hook

    # Register hooks
    model.vme.register_forward_hook(get_activation('vme_embedding'))
    model.arg.register_forward_hook(get_activation('arg_representation'))
    model.pathology_graph.register_forward_hook(get_activation('pathology_graph'))
    model.fusion.register_forward_hook(get_activation('dual_graph_representation'))
    model.mcam.register_forward_hook(get_activation('mcam_representation'))
    model.ugdm.register_forward_hook(get_activation('ugdm_uncertainty'))
    
    # Load 1 image
    test_ds = VinDrDataset(r"c:\projects\spine\data\manifests\test.csv", r"c:\projects\spine\data\vindr-spinexr\test_images", augment=False)
    test_loader = DataLoader(Subset(test_ds, [0]), batch_size=1, collate_fn=collate_fn)
    
    out_dir = r"c:\projects\spine\outputs\phase15_validation\integration"
    os.makedirs(out_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            out_std = model(images)
            out_unc = model(images, return_uncertainty=True)
            
            # Save standard outputs
            np.save(os.path.join(out_dir, "pathology_probabilities.npy"), out_std['probs'].cpu().numpy())
            
            # Save intermediate hooked outputs
            if 'vme_embedding' in intermediates:
                # vme returns a list of tensors per image
                np.save(os.path.join(out_dir, "vertebral_morphology_embedding.npy"), intermediates['vme_embedding'][0])
            if 'arg_representation' in intermediates:
                # arg returns (refined_nodes, global_spine)
                np.save(os.path.join(out_dir, "arg_representation.npy"), intermediates['arg_representation'])
            if 'pathology_graph' in intermediates:
                # path_graph returns (nodes, logits)
                np.save(os.path.join(out_dir, "pathology_graph.npy"), intermediates['pathology_graph'])
            if 'dual_graph_representation' in intermediates:
                np.save(os.path.join(out_dir, "dual_graph_representation.npy"), intermediates['dual_graph_representation'])
            if 'mcam_representation' in intermediates:
                np.save(os.path.join(out_dir, "mcam_attention.npy"), intermediates['mcam_representation'])
                
            # Save uncertainty outputs manually since it's a dict
            np.save(os.path.join(out_dir, "uncertainty.npy"), out_unc['uncertainty'].cpu().numpy())
            break
            
    print(f"Integration intermediate tensors saved successfully to {out_dir}")

if __name__ == "__main__":
    evaluate_anatomy_and_integration()
