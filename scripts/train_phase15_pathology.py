import os
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from src.data.dataset import VinDrDataset, collate_fn
from src.models.neuromorphnet_final import AnatomyConstrainedDualGraphNeuroMorphNet
import time

def train_pathology():
    print("Part C: Training the Pathology Component of Final NeuroMorphNet")
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    out_dir = r"c:\projects\spine\outputs\phase15_validation\checkpoints"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Datasets
    # We use a 100-image subset for train and 20 for val to simulate the training loop 
    # without requiring hours of computation for this architectural validation step.
    full_train = VinDrDataset(r"c:\projects\spine\data\manifests\train.csv", r"c:\projects\spine\data\vindr-spinexr\train_images", augment=True)
    full_val = VinDrDataset(r"c:\projects\spine\data\manifests\test.csv", r"c:\projects\spine\data\vindr-spinexr\test_images", augment=False)
    
    train_subset = Subset(full_train, list(range(100)))
    val_subset = Subset(full_val, list(range(20)))
    
    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # 2. Model
    model = AnatomyConstrainedDualGraphNeuroMorphNet(embed_dim=256, num_pathologies=8).to(device)
    
    # The localizer is frozen because we lack sufficient anatomical training data (Part B)
    for param in model.localizer.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    best_epoch = 0
    num_epochs = 2
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets_tensor = torch.stack([t['labels'] for t in targets]).to(device)
            
            optimizer.zero_grad()
            with autocast():
                out = model(images, targets=targets_tensor)
                loss = out['loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets_tensor = torch.stack([t['labels'] for t in targets]).to(device)
                
                with autocast():
                    out = model(images, targets=targets_tensor)
                    loss = out['loss']
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(out_dir, "best_pathology_model.pt"))
            
    print(f"Training completed in {time.time() - start_time:.2f}s")
    print(f"Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")
    
    # Write summary to a log file for audit
    with open(os.path.join(out_dir, "training_summary.txt"), "w") as f:
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Checkpoint: best_pathology_model.pt\n")

if __name__ == "__main__":
    train_pathology()
