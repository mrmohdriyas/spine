import torch
import torch.nn as nn

class UncertaintyGuidedDecisionModule(nn.Module):
    """
    Part J: Uncertainty-Guided Decision Module (UGDM).
    Uses Monte Carlo (MC) Dropout to estimate epistemic uncertainty
    for each pathology prediction.
    """
    def __init__(self, num_mc_passes=10):
        super().__init__()
        self.num_mc_passes = num_mc_passes
        
    def enable_dropout(self, model):
        """Forces dropout layers to be active even in eval mode."""
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
    def forward(self, head_module, x):
        """
        Args:
            head_module: The MultiPathologyHead module.
            x: Tensor [B, embed_dim] (the features to classify).
            
        Returns:
            dict containing:
                - 'mean_prob': [B, 8]
                - 'variance': [B, 8]
                - 'uncertainty': [B, 8] (in this case, just variance or standard deviation)
                - 'confidence': [B, 8] (1 - uncertainty normalized)
        """
        # Save original mode
        was_training = head_module.training
        
        # Enable dropout specifically
        self.enable_dropout(head_module)
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.num_mc_passes):
                logits, _ = head_module(x)
                probs = torch.sigmoid(logits)
                predictions.append(probs.unsqueeze(0))
                
        # Stack predictions: [num_passes, B, 8]
        predictions = torch.cat(predictions, dim=0)
        
        # Calculate statistics
        mean_prob = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        # We define uncertainty simply as the variance of MC Dropout predictions.
        # Confidence can be loosely defined as the maximum margin or 1 - normalized variance.
        # For simplicity:
        uncertainty = variance
        
        # Max variance of a Bernoulli is 0.25 (at p=0.5). So normalize by 0.25 to get [0, 1] scale.
        normalized_variance = torch.clamp(variance / 0.25, 0, 1)
        confidence = 1.0 - normalized_variance
        
        # Restore original mode
        head_module.train(was_training)
        
        return {
            "mean_prob": mean_prob,
            "variance": variance,
            "uncertainty": uncertainty,
            "confidence": confidence
        }
