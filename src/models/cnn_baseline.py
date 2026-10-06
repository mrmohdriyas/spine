import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CNNBaseline(nn.Module):
    def __init__(self, num_classes=8):
        super(CNNBaseline, self).__init__()
        # Load pretrained ResNet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Replace the classifier head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x is (B, 3, H, W)
        return self.model(x)
