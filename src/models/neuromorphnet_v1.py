import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .morphology_encoder import MorphologyEncoder

class NeuromorphnetV1(nn.Module):
    def __init__(self, num_classes=8, morphology_dim=8, morphology_embed_dim=64):
        super(NeuromorphnetV1, self).__init__()
        
        # Load pretrained ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the classification head (fc) to get 512D features
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Morphology Encoder
        self.morphology_encoder = MorphologyEncoder(input_dim=morphology_dim, output_dim=morphology_embed_dim)
        
        # Feature Fusion
        in_features = resnet.fc.in_features  # typically 512
        fusion_dim = in_features + morphology_embed_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image, morphology):
        # Image features
        # image is (B, 3, H, W)
        img_feats = self.image_encoder(image)  # (B, 512, 1, 1)
        img_feats = torch.flatten(img_feats, 1) # (B, 512)
        
        # Morphology features
        # morphology is (B, 8)
        morph_feats = self.morphology_encoder(morphology) # (B, 64)
        
        # Fusion
        fused = torch.cat([img_feats, morph_feats], dim=1) # (B, 576)
        logits = self.fusion(fused) # (B, 8)
        
        return logits
