import sys
import inspect
import torch

sys.path.insert(0, r"c:\projects\spine")
from src.dataset.spine_dataset import SpineDataset
from src.models.neuromorphnet_v3 import NeuromorphnetV3
from src.models.v3_nomorph import V3NoMorph

def verify_leakage_free():
    print("Verifying NeuroMorphNet V3 pipeline for data leakage...")
    
    # 1. Check SpineDataset returns exactly 2 items
    ds = SpineDataset(r"c:\projects\spine\data\manifests\val.csv", 
                      r"c:\projects\spine\data\class_mapping.json", 
                      transforms=None, return_morphology=False, return_graph=False)
                      
    item = ds[0]
    if len(item) != 2:
        print(f"FAILED: Dataset returned {len(item)} items instead of 2. Possible leakage.")
        sys.exit(1)
    else:
        print("PASS: Dataset returns only image and target.")
        
    image, target = item
    if not isinstance(target, torch.Tensor):
        print("FAILED: Target is not a tensor.")
        sys.exit(1)
        
    if not isinstance(image, torch.Tensor):
        import numpy as np
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        else:
            print("FAILED: Dataset image is neither tensor nor numpy array.")
            sys.exit(1)
        
    # 2. Check Model Forward Signature
    models_to_test = [NeuromorphnetV3(num_classes=8), V3NoMorph(num_classes=8)]
    
    for model in models_to_test:
        model_name = model.__class__.__name__
        sig = inspect.signature(model.forward)
        params = list(sig.parameters.keys())
        
        if len(params) != 1 or params[0] != 'images':
            print(f"FAILED: {model_name} forward signature accepts: {params}. It should only accept 'images'.")
            sys.exit(1)
        else:
            print(f"PASS: {model_name} forward signature accepts only 'images'.")
            
        # 3. Check Forward Pass
        try:
            # Create a dummy batch of 2 images
            dummy_images = torch.stack([image.unsqueeze(0).repeat(3, 1, 1), image.unsqueeze(0).repeat(3, 1, 1)])
            # resize for resnet
            import torch.nn.functional as F
            dummy_images = F.interpolate(dummy_images, size=(224, 224))
            
            logits = model(dummy_images)
            if logits.shape != (2, 8):
                print(f"FAILED: {model_name} expected logits shape (2, 8), got {logits.shape}.")
                sys.exit(1)
            print(f"PASS: {model_name} Forward pass completed successfully using only images.")
        except Exception as e:
            print(f"FAILED: {model_name} Forward pass crashed: {e}")
            sys.exit(1)
        
    print("\nVERIFICATION SUCCESSFUL: No annotation leakage detected in the model input pipeline.")
    
if __name__ == '__main__':
    verify_leakage_free()
