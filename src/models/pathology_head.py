import torch
import torch.nn as nn

class MultiPathologyHead(nn.Module):
    """
    Part I: Multi-Pathology Prediction Head
    Takes the final context-aware embedding from MCAM and outputs 8 logits
    for the VinDr pathologies.
    """
    def __init__(self, embed_dim=256, num_classes=8):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, x, targets=None):
        """
        Args:
            x: Tensor [B, embed_dim]
            targets: Optional Tensor [B, 8] containing multi-label ground truth (0 or 1).
            
        Returns:
            logits: Tensor [B, 8]
            loss: scalar loss (if targets provided)
        """
        logits = self.classifier(x)
        
        loss = None
        if targets is not None:
            loss = self.criterion(logits, targets.float())
            
        return logits, loss
