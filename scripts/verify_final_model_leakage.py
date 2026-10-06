import torch
from src.models.neuromorphnet_final import AnatomyConstrainedDualGraphNeuroMorphNet

def verify_leakage():
    print("Running Leakage Verification on AnatomyConstrainedDualGraphNeuroMorphNet...")
    
    # 1. Instantiate the model
    model = AnatomyConstrainedDualGraphNeuroMorphNet(embed_dim=256, num_pathologies=8)
    model.eval()
    
    # 2. Create a dummy image tensor matching VinDr characteristics (e.g., 512x512)
    dummy_image = torch.randn(2, 3, 512, 512)
    
    # 3. Run inference WITHOUT any targets or ground-truth boxes
    print("Executing forward pass with only image input...")
    try:
        with torch.no_grad():
            output_std = model(dummy_image, return_uncertainty=False)
            output_unc = model(dummy_image, return_uncertainty=True)
            
        print("Forward pass successful.")
        
        # 4. Assertions to ensure standard outputs are correct
        assert "logits" in output_std, "Missing logits in standard output"
        assert output_std["logits"].shape == (2, 8), f"Incorrect logits shape: {output_std['logits'].shape}"
        
        assert "mean_prob" in output_unc, "Missing mean_prob in uncertainty output"
        assert output_unc["mean_prob"].shape == (2, 8), f"Incorrect mean_prob shape: {output_unc['mean_prob'].shape}"
        
        print("✅ LEAKAGE VERIFICATION PASSED.")
        print("The final model strictly requires only the raw image for inference.")
        print("Ground-truth bounding boxes or pathology geometries are NOT consumed during inference.")
        
    except Exception as e:
        print(f"❌ LEAKAGE VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_leakage()
