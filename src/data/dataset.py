import os
import torch
import numpy as np
import json
from src.dataset.spine_dataset import SpineDataset
from torchvision import transforms

class VinDrDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, image_dir, augment=False):
        mapping_path = r"c:\projects\spine\data\class_mapping.json"
        if not os.path.exists(mapping_path):
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            with open(mapping_path, 'w') as f:
                json.dump({
                    "0": "Osteophytes", "1": "No finding", "2": "Disc space narrowing",
                    "3": "Other lesions", "4": "Surgical implant", "5": "Foraminal stenosis",
                    "6": "Spondylolysthesis", "7": "Vertebral collapse"
                }, f)
                
        self.dataset = SpineDataset(manifest_path, mapping_path)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # SpineDataset returns numpy array H,W and target tensor
        img_np, target_vec = self.dataset[idx]
        
        # Ensure it is float32
        img_np = img_np.astype(np.float32)
        
        # Convert to tensor [C, H, W]
        img_tensor = torch.from_numpy(img_np)
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)
            
        img_tensor = img_tensor.repeat(3, 1, 1) # [3, H, W]
        
        # Resize to 512x512
        img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
        
        return img_tensor, {'labels': target_vec}

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets
