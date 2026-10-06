import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class VertebralLocalizer(nn.Module):
    """
    Part C: Trainable Anatomical Localization Module.
    Lightweight localizer suitable for RTX 4050 (6GB).
    Outputs: Box coordinates, Confidence, Vertebral Identity, Visibility.
    Classes: 0: Background, 1: L1, 2: L2, 3: L3, 4: L4, 5: L5
    """
    def __init__(self, num_classes=6, pretrained_backbone=False):
        super().__init__()
        # Using mobilenet_v3_large_fpn for lightweight efficiency on RTX 4050
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=None, weights_backbone=None
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    def forward(self, images, targets=None):
        """
        In Training: Returns loss dict (L_box, L_objectness, etc).
        In Inference: Returns list of dicts with 'boxes', 'labels', 'scores'.
        """
        # The internal Faster R-CNN already handles L_box, L_objectness, and L_identity (cls_loss).
        # Visibility can be inferred from score/presence, as explicit visibility 
        # classification requires a custom ROI head which is beyond standard Faster R-CNN.
        return self.model(images, targets)
        
    def extract_regions(self, images, conf_threshold=0.5):
        """
        Helper for inference to output ordered L1-L5 regions.
        Returns:
            list of dicts per image:
            {
                "boxes": [N, 4],
                "labels": [N],  # 1=L1, 2=L2, etc.
                "scores": [N],
                "visibility": [N] # boolean or pseudo-conf
            }
        """
        self.eval()
        with torch.no_grad():
            preds = self.model(images)
            
        results = []
        for p in preds:
            keep = p['scores'] > conf_threshold
            b = p['boxes'][keep]
            l = p['labels'][keep]
            s = p['scores'][keep]
            
            # Simple heuristic for visibility: if score > threshold, it's visible.
            # In a fully supervised visibility dataset, this would be a separate branch.
            v = (s > conf_threshold).float()
            
            results.append({
                "boxes": b,
                "labels": l,
                "scores": s,
                "visibility": v
            })
        return results
