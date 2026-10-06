import os
import torch
import torch.nn as nn
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from models.dann_faster_rcnn import DANN_FasterRCNN

class DRRDatasetV2(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.imgs = list(sorted(os.listdir(os.path.join(root, split))))
        self.df = pd.read_csv(os.path.join(root, "annotations", "drr_annotations.csv"))
        
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, self.split, img_name)
        scan_id = img_name.split('.')[0]
        
        img = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(img)
        
        scan_df = self.df[self.df["scan_id"] == scan_id]
        
        boxes = []
        labels = []
        
        for _, row in scan_df.iterrows():
            xmin = max(0, row['x_min'])
            ymin = max(0, row['y_min'])
            xmax = min(img.width, row['x_max'])
            ymax = min(img.height, row['y_max'])
            
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(row['label_id']) - 19)
                
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        return img_tensor, target
        
    def __len__(self):
        return len(self.imgs)

class VinDrUnlabeledDataset(Dataset):
    def __init__(self, manifest_path, img_dir, num_samples=100):
        df = pd.read_csv(manifest_path)
        self.img_ids = df['image_id'].unique()[:num_samples]
        self.img_dir = img_dir
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        dcm_path = os.path.join(self.img_dir, f"{img_id}.dicom")
        
        ds = pydicom.dcmread(dcm_path)
        img_array = ds.pixel_array.astype(float)
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255
        img = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")
        img = img.resize((256, 256)) # resize to roughly match DRR scale for easier batches
        
        img_tensor = torchvision.transforms.functional.to_tensor(img)
        return img_tensor
        
    def __len__(self):
        return len(self.img_ids)

def collate_fn(batch):
    return tuple(zip(*batch))

def collate_fn_vindr(batch):
    return batch

def train_dann():
    out_dir = r"c:\projects\spine\outputs\phase12_domain_adaptation"
    os.makedirs(out_dir, exist_ok=True)
    
    source_dataset = DRRDatasetV2(r"c:\projects\spine\data\verse_drr_v2", "train")
    source_loader = DataLoader(source_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    target_dataset = VinDrUnlabeledDataset(
        r"c:\projects\spine\data\manifests\train.csv",
        r"C:\Users\Mohamed Riyas\.cache\kagglehub\datasets\siddhale937e92739\annotated-medical-image-dataset-for-spinal-lesions\versions\1\physionet.org\files\vindr-spinexr\1.0.0\train_images",
        num_samples=100
    )
    target_loader = DataLoader(target_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_vindr)
    
    model = DANN_FasterRCNN(num_classes=6)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    bce_loss = nn.BCEWithLogitsLoss()
    
    num_epochs = 10
    lambda_domain = 0.1
    
    history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_det_loss = 0
        epoch_dom_loss = 0
        correct_domain = 0
        total_domain = 0
        
        target_iter = iter(target_loader)
        
        for source_images, source_targets in source_loader:
            optimizer.zero_grad()
            
            # --- Source Forward ---
            source_images = list(img.to(device) for img in source_images)
            source_targets = [{k: v.to(device) for k, v in t.items()} for t in source_targets]
            
            det_losses, source_domain_logits = model(source_images, source_targets, alpha=1.0, is_target_domain=False)
            det_loss_total = sum(loss for loss in det_losses.values())
            
            # Source Domain Label = 0
            source_domain_labels = torch.zeros_like(source_domain_logits)
            source_domain_loss = bce_loss(source_domain_logits, source_domain_labels)
            
            # Accuracy metric
            source_preds = torch.sigmoid(source_domain_logits) < 0.5
            correct_domain += source_preds.sum().item()
            total_domain += source_domain_logits.shape[0]
            
            # --- Target Forward ---
            try:
                target_images = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_images = next(target_iter)
                
            target_images = list(img.to(device) for img in target_images)
            
            _, target_domain_logits = model(target_images, alpha=1.0, is_target_domain=True)
            
            # Target Domain Label = 1
            target_domain_labels = torch.ones_like(target_domain_logits)
            target_domain_loss = bce_loss(target_domain_logits, target_domain_labels)
            
            target_preds = torch.sigmoid(target_domain_logits) >= 0.5
            correct_domain += target_preds.sum().item()
            total_domain += target_domain_logits.shape[0]
            
            # --- Total Loss ---
            domain_loss_total = (source_domain_loss + target_domain_loss) / 2.0
            total_loss = det_loss_total + lambda_domain * domain_loss_total
            
            total_loss.backward()
            optimizer.step()
            
            epoch_det_loss += det_loss_total.item()
            epoch_dom_loss += domain_loss_total.item()
            
        domain_acc = correct_domain / max(total_domain, 1)
        print(f"Epoch {epoch+1}/{num_epochs} | Det Loss: {epoch_det_loss/len(source_loader):.4f} | Dom Loss: {epoch_dom_loss/len(source_loader):.4f} | Dom Acc: {domain_acc:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "det_loss": epoch_det_loss/len(source_loader),
            "dom_loss": epoch_dom_loss/len(source_loader),
            "dom_acc": domain_acc
        })
        
    torch.save(model.faster_rcnn.state_dict(), os.path.join(out_dir, "detector_dann.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "domain_classifier_history.csv"), index=False)
    print("DANN Training Complete.")

if __name__ == "__main__":
    train_dann()
