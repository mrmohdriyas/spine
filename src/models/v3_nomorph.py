import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class V3NoMorph(nn.Module):
    def __init__(self, num_classes=8):
        super(V3NoMorph, self).__init__()
        
        # Load pretrained ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # We need the feature map before average pooling, matching V3 exactly.
        # In ResNet18, layer4 output is typically (B, 512, 7, 7)
        modules = list(resnet.children())[:-2] # Remove AdaptiveAvgPool2d and Linear
        self.backbone = nn.Sequential(*modules)
        
        # Image Representation Branch
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head (matching V3's fusion depth but without morphology input)
        # V3 used Linear(576, 128) -> ReLU -> Linear(128, 8). 
        # V3-NoMorph uses Linear(512, 128) -> ReLU -> Linear(128, 8).
        self.classification_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, images):
        """
        images: (B, 3, H, W)
        """
        # Feature map (B, 512, 7, 7)
        feature_map = self.backbone(images)
        
        # Image Representation (B, 512)
        img_feats = self.image_pool(feature_map)
        img_feats = torch.flatten(img_feats, 1)
        
        # Classification
        logits = self.classification_head(img_feats) # (B, 8)
        
        return logits
