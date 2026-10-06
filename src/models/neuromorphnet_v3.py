import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class NeuromorphnetV3(nn.Module):
    def __init__(self, num_classes=8):
        super(NeuromorphnetV3, self).__init__()
        
        # Load pretrained ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # We need the feature map before average pooling. 
        # In ResNet18, layer4 output is typically (B, 512, 7, 7)
        modules = list(resnet.children())[:-2] # Remove AdaptiveAvgPool2d and Linear
        self.backbone = nn.Sequential(*modules)
        
        # Image Representation Branch
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Image-Derived Morphology Branch
        self.morphology_branch = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion
        # 512 from image branch + 64 from morphology branch = 576
        self.fusion = nn.Sequential(
            nn.Linear(576, 128),
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
        
        # Morphology Representation (B, 64)
        morph_feats = self.morphology_branch(feature_map)
        
        # Fusion
        fused = torch.cat([img_feats, morph_feats], dim=1) # (B, 576)
        logits = self.fusion(fused) # (B, 8)
        
        return logits
