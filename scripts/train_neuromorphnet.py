import os
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import VinDrDataset, collate_fn
from src.models.neuromorphnet_final import AnatomyConstrainedDualGraphNeuroMorphNet

def train_staged_prototype():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    out_dir = r"c:\projects\spine\outputs\phase14_neuromorphnet\checkpoints"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Dataset
    train_dataset = VinDrDataset(r"c:\projects\spine\data\manifests\train.csv", r"c:\projects\spine\data\vindr-spinexr\train_images", augment=True)
    # Subset for prototype execution (just to prove the architecture runs end-to-end)
    subset_indices = list(range(32))
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # 2. Model
    model = AnatomyConstrainedDualGraphNeuroMorphNet(embed_dim=256, num_pathologies=8)
    
    # Optionally load Stage 1 (Localizer) weights from Phase 13 if they existed and were helpful,
    # but we proved they were 0 precision. We just start from scratch/pretrained backbone.
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()
    
    print("Starting Stage 7: Complete End-to-End Prototype Training (Mixed Precision)")
    
    model.train()
    # To prevent issues with Localizer producing 0 boxes and breaking graphs,
    # we rely on the internal fallback mechanisms (e.g., returning identity tensors) implemented in the modules.
    
    num_epochs = 2
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets_tensor = torch.stack([t['labels'] for t in targets]).to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                out = model(images, targets=targets_tensor)
                loss = out['loss']
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), os.path.join(out_dir, "neuromorphnet_final_prototype.pt"))
    print(f"Checkpoint saved to {out_dir}")

if __name__ == "__main__":
    train_staged_prototype()
