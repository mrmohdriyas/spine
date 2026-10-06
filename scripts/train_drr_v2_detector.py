import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        return img_tensor, target
        
    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))

def train_v2_detector():
    dataset = DRRDatasetV2(r"c:\projects\spine\data\verse_drr_v2", "train")
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    num_classes = 6
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
        print(f"V2 Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader):.4f}")
        
    torch.save(model.state_dict(), r"c:\projects\spine\outputs\phase11_domain_adaptation\detector_v2.pt")
    print("V2 Detector training complete.")

if __name__ == "__main__":
    train_v2_detector()
