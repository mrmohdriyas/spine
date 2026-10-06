import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .morphology_encoder import MorphologyEncoder
from .spatial_graph_encoder import SpatialGraphEncoder

class NeuromorphnetV2(nn.Module):
    def __init__(self, num_classes=8, morphology_dim=8, morphology_embed_dim=64, graph_embed_dim=64):
        super(NeuromorphnetV2, self).__init__()
        
        # Load pretrained ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the classification head (fc) to get 512D features
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Morphology Encoder
        self.morphology_encoder = MorphologyEncoder(input_dim=morphology_dim, output_dim=morphology_embed_dim)
        
        # Spatial Graph Encoder
        self.graph_encoder = SpatialGraphEncoder(node_dim=6, edge_dim=5, embed_dim=graph_embed_dim)
        
        # Feature Fusion
        in_features = resnet.fc.in_features  # typically 512
        fusion_dim = in_features + morphology_embed_dim + graph_embed_dim # 512 + 64 + 64 = 640
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, images, morphologies, graphs):
        """
        images: (B, 3, H, W)
        morphologies: (B, 8)
        graphs: List of B graph dictionaries
        """
        # Image features
        img_feats = self.image_encoder(images)  # (B, 512, 1, 1)
        img_feats = torch.flatten(img_feats, 1) # (B, 512)
        
        # Morphology features
        morph_feats = self.morphology_encoder(morphologies) # (B, 64)
        
        # Graph features
        graph_feats_list = []
        for g in graphs:
            # Process each graph individually
            g_emb = self.graph_encoder(g) # (1, 64)
            graph_feats_list.append(g_emb)
            
        graph_feats = torch.cat(graph_feats_list, dim=0) # (B, 64)
        
        # Fusion
        fused = torch.cat([img_feats, morph_feats, graph_feats], dim=1) # (B, 640)
        logits = self.fusion(fused) # (B, 8)
        
        return logits
