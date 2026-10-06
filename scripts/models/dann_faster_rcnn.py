import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN_FasterRCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.faster_rcnn = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Load Phase 11 checkpoint to start with a good source representation
        ckpt = torch.load(r"c:\projects\spine\outputs\phase11_domain_adaptation\detector_v2.pt", weights_only=True)
        self.faster_rcnn.load_state_dict(ckpt)
        
        # Domain classifier on FPN pooled features (256 channels)
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 1) # binary: 0 = Source (VerSe), 1 = Target (VinDr)
        )
        
    def forward(self, images, targets=None, alpha=1.0, is_target_domain=False):
        """
        If is_target_domain is True, we ONLY calculate the domain loss (no bounding box targets required).
        If is_target_domain is False, we calculate both bbox losses and domain loss.
        """
        if self.training:
            original_image_sizes = []
            for img in images:
                val = img.shape[-2:]
                original_image_sizes.append((val[0], val[1]))

            images_transformed, targets_transformed = self.faster_rcnn.transform(images, targets)
            features = self.faster_rcnn.backbone(images_transformed.tensors)
            
            # FPN outputs typically include 'pool' for the highest level features
            if 'pool' in features:
                pool_feat = features['pool']
            else:
                # fallback to the coarsest feature map
                pool_feat = list(features.values())[-1]
            
            # Global Average Pooling
            gap = torch.mean(pool_feat, dim=[2, 3]) # (N, 256)
            
            # Apply GRL
            reverse_feat = GRL.apply(gap, alpha)
            
            # Domain Prediction
            domain_logits = self.domain_classifier(reverse_feat)
            
            losses = {}
            if not is_target_domain:
                # Compute standard Faster R-CNN losses for source domain
                proposals, proposal_losses = self.faster_rcnn.rpn(images_transformed, features, targets_transformed)
                detections, detector_losses = self.faster_rcnn.roi_heads(features, proposals, images_transformed.image_sizes, targets_transformed)
                losses.update(detector_losses)
                losses.update(proposal_losses)
            
            return losses, domain_logits
        else:
            return self.faster_rcnn(images)
