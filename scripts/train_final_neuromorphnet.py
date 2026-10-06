import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from src.data.dataset import VinDrDataset, collate_fn
from src.models.neuromorphnet_final import AnatomyConstrainedDualGraphNeuroMorphNet

def train():
    print("Starting Final NeuroMorphNet Training...")
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    out_dir = r"c:\projects\spine\outputs\final_neuromorphnet"
    os.makedirs(out_dir, exist_ok=True)
    
    torch.manual_seed(42)
    
    # 1. Dataset
    print("Loading datasets...")
    full_train = VinDrDataset(r"c:\projects\spine\data\manifests\train.csv", r"c:\projects\spine\data\vindr-spinexr\train_images", augment=True)
    full_val = VinDrDataset(r"c:\projects\spine\data\manifests\test.csv", r"c:\projects\spine\data\vindr-spinexr\test_images", augment=False)
    
    # Using small subsets due to hardware constraints (6GB VRAM, no background long-running task)
    train_subset = Subset(full_train, list(range(100)))
    val_subset = Subset(full_val, list(range(25)))
    
    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # 2. Model
    print("Initializing model...")
    model = AnatomyConstrainedDualGraphNeuroMorphNet(embed_dim=256, num_pathologies=8).to(device)
    
    # We freeze the localizer because there is no anatomical supervision available to train it properly.
    for param in model.localizer.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = GradScaler('cuda')
    
    num_epochs = 2
    best_val_loss = float('inf')
    best_epoch = 0
    history = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            targets_tensor = torch.stack([t['labels'] for t in targets]).to(device)
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda'):
                out = model(images, targets=targets_tensor)
                loss = out['loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                targets_tensor = torch.stack([t['labels'] for t in targets]).to(device)
                
                with torch.autocast(device_type='cuda'):
                    out = model(images, targets=targets_tensor)
                    loss = out['loss']
                    
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr}")
        history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
            
    total_time = time.time() - start_time
    
    print("\n--- Training Complete ---")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Training Time: {total_time:.2f} seconds")
    if torch.cuda.is_available():
        print(f"GPU Used: {torch.cuda.get_device_name(0)}")
        
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

if __name__ == "__main__":
    train()
