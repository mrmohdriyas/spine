import torch
import torch.nn as nn
import torchvision
from torchvision.ops import roi_align

class VertebralMorphologyEncoder(nn.Module):
    """
    Part D: Vertebral Morphology Encoder (VME).
    Extracts morphology features (shape, texture, aspect ratio) from 
    the cropped image region of each localized vertebra.
    """
    def __init__(self, feature_dim=256, roi_size=(7, 7)):
        super().__init__()
        self.roi_size = roi_size
        
        # Lightweight CNN feature extractor (e.g., truncated ResNet18)
        resnet = torchvision.models.resnet18(weights=None)
        # Use up to layer3 (stride 16, 256 channels)
        self.feature_extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        self.spatial_scale = 1.0 / 16.0 # Layer 3 stride is 16
        
        # Flatten and embed the ROI features + geometric features
        # Geometric features: width, height, aspect ratio (3 dims)
        cnn_out_dim = 256 * roi_size[0] * roi_size[1]
        self.embedding = nn.Sequential(
            nn.Linear(cnn_out_dim + 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, images, boxes_list):
        """
        images: Tensor [B, C, H, W]
        boxes_list: List of Tensors [N_i, 4] containing vertebral boxes
        
        Returns:
            List of embeddings [N_i, feature_dim] per image.
        """
        B = images.size(0)
        
        # 1. Extract global image features
        features = self.feature_extractor(images)
        
        # 2. Extract ROI features for each box
        # roi_align requires boxes as a list of tensors [N, 4]
        # It returns [sum(N), C, roi_size, roi_size]
        if sum(len(b) for b in boxes_list) == 0:
            return [torch.empty(0, self.embedding[-1].out_features, device=images.device) for _ in range(B)]
            
        roi_features = roi_align(features, boxes_list, output_size=self.roi_size, spatial_scale=self.spatial_scale)
        roi_features = roi_features.flatten(start_dim=1) # [sum(N), C*H*W]
        
        # 3. Calculate explicit geometric features (width, height, aspect ratio)
        geom_features = []
        for boxes in boxes_list:
            if len(boxes) > 0:
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                ar = w / (h + 1e-6)
                # Normalize arbitrarily by image size to keep scale reasonable
                w_norm = w / images.size(3)
                h_norm = h / images.size(2)
                geom = torch.stack([w_norm, h_norm, ar], dim=1)
                geom_features.append(geom)
            
        if len(geom_features) > 0:
            geom_features = torch.cat(geom_features, dim=0) # [sum(N), 3]
        else:
            geom_features = torch.empty((0, 3), device=images.device)
            
        # 4. Concatenate and embed
        combined_features = torch.cat([roi_features, geom_features], dim=1)
        embeddings = self.embedding(combined_features) # [sum(N), feature_dim]
        
        # 5. Split back into list per image
        split_sizes = [len(b) for b in boxes_list]
        return list(torch.split(embeddings, split_sizes))
