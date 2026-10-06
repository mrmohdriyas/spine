import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

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

def calculate_pos_weight(dataset):
    print("Calculating positive weights from training data...")
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

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = {
        "batch_size": 8,
        "epochs": 10,
        "lr": 1e-4,
        "random_seed": 42
    }
    
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    
    tf = BasicTransform(size=(224, 224))
    
    train_ds = SpineDataset(r"c:\projects\spine\data\manifests\train.csv", 
                            r"c:\projects\spine\data\class_mapping.json", 
                            transforms=tf, return_morphology=True, return_graph=True)
                            
    val_ds = SpineDataset(r"c:\projects\spine\data\manifests\val.csv", 
                          r"c:\projects\spine\data\class_mapping.json", 
                          transforms=tf, return_morphology=True, return_graph=True)
                          
    print(f"Total unique training images: {len(train_ds)}")
    print(f"Total unique validation images: {len(val_ds)}")
    
    pos_weight = calculate_pos_weight(train_ds).to(device)
    print(f"Class pos_weights: {pos_weight}")
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, 
                              num_workers=4, persistent_workers=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, 
                            num_workers=4, persistent_workers=True, collate_fn=collate_fn)
    
    model = NeuromorphnetV2(num_classes=train_ds.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    history = {
        "train_loss": [],
        "val_loss": []
    }
    
    best_val_loss = float('inf')
    output_dir = r"c:\projects\spine\outputs\neuromorphnet_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        for batch_idx, (images, targets, morphs, graphs) in enumerate(tqdm(train_loader, desc="Training")):
            images = images.to(device)
            targets = targets.to(device)
            morphs = morphs.to(device)
            
            # move graphs to device
            for g in graphs:
                g['nodes'] = g['nodes'].to(device)
                g['edge_indices'] = g['edge_indices'].to(device)
                g['edge_features'] = g['edge_features'].to(device)
                
            optimizer.zero_grad()
            logits = model(images, morphs, graphs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, targets, morphs, graphs) in enumerate(tqdm(val_loader, desc="Validation")):
                images = images.to(device)
                targets = targets.to(device)
                morphs = morphs.to(device)
                
                for g in graphs:
                    g['nodes'] = g['nodes'].to(device)
                    g['edge_indices'] = g['edge_indices'].to(device)
                    g['edge_features'] = g['edge_features'].to(device)
                    
                logits = model(images, morphs, graphs)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            
    with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=4)
        
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == '__main__':
    train()
