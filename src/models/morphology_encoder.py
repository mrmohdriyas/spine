import torch
import torch.nn as nn

class MorphologyEncoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32, output_dim=64):
        super(MorphologyEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x is (B, 8)
        return self.mlp(x)
