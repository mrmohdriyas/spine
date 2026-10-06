import os
import torch
import json
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
from models.dann_faster_rcnn import DANN_FasterRCNN
from evaluate_real_xray import evaluate_models

class RealXrayDataset(Dataset):
    def __init__(self, json_path, img_dir):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.label_map = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, f"{item['image_id']}.png")
        img = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(img)
        
        boxes = []
        labels = []
        for ann in item['annotations']:
            if ann['vertebra'] in self.label_map:
                boxes.append(ann['bbox'])
                labels.append(self.label_map[ann['vertebra']])
                
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

def collate_fn(batch):
    return tuple(zip(*batch))

def finetune():
    data_dir = r"c:\projects\spine\data\vindr_vertebral_annotations"
    out_dir = r"c:\projects\spine\outputs\phase13_real_xray_validation"
    os.makedirs(out_dir, exist_ok=True)
    
    train_dataset = RealXrayDataset(os.path.join(data_dir, "train.json"), os.path.join(data_dir, "images"))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    model = DANN_FasterRCNN(num_classes=6)
    model.faster_rcnn.load_state_dict(torch.load(r"c:\projects\spine\outputs\phase12_domain_adaptation\detector_dann.pt", weights_only=True))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    # Only fine-tune the box predictor to avoid catastrophic forgetting of feature extraction
    params = [p for p in model.faster_rcnn.roi_heads.box_predictor.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    num_epochs = 20
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            # Standard Faster R-CNN forward pass for fine-tuning
            loss_dict = model.faster_rcnn(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
        
    ft_model_path = os.path.join(out_dir, "detector_dann_finetuned.pt")
    torch.save(model.faster_rcnn.state_dict(), ft_model_path)
    print("Fine-tuning complete.")
    
    # Evaluate Fine-tuned model alongside others
    model.eval()
    evaluate_models(device, os.path.join(data_dir, "test.json"), os.path.join(out_dir, "model_comparison.csv"), ft_model=model.faster_rcnn)

if __name__ == "__main__":
    finetune()
