import torch
import torch.nn as nn

class MorphologyAwareCrossAttention(nn.Module):
    """
    Part H: Morphology-Aware Cross-Attention Module (MCAM).
    The fused Dual-Graph pathology nodes act as Queries.
    They attend to the high-resolution raw image features and 
    the raw vertebral morphology features (Keys/Values) to ground 
    their final predictions in precise visual and shape details.
    """
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # We need a small pooling mechanism to collapse the G*G nodes into a single feature vector per image
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, dual_graph_features, image_features, vme_features_list):
        """
        Args:
            dual_graph_features: Tensor [B, G*G, embed_dim]
            image_features: Tensor [B, C, H, W]
            vme_features_list: List of Tensors [N_i, embed_dim] for B images
            
        Returns:
            final_embeddings: Tensor [B, embed_dim] ready for classification
        """
        B, C, H, W = image_features.shape
        
        # Flatten image features: [B, H*W, C]
        img_seq = image_features.view(B, C, -1).transpose(1, 2)
        
        out_features = []
        
        for i in range(B):
            query = dual_graph_features[i:i+1] # [1, G*G, embed_dim]
            img_kv = img_seq[i:i+1] # [1, H*W, embed_dim]
            vme_kv = vme_features_list[i].unsqueeze(0) # [1, N_i, embed_dim]
            
            # Combine Image and Morphology features into a single Key/Value memory bank
            if vme_kv.size(1) > 0:
                kv = torch.cat([img_kv, vme_kv], dim=1)
            else:
                kv = img_kv
                
            attn_out, _ = self.cross_attn(query=query, key=kv, value=kv)
            
            out1 = self.norm1(query + attn_out)
            out2 = self.ffn(out1)
            out = self.norm2(out1 + out2) # [1, G*G, embed_dim]
            
            # Collapse the G*G nodes into a single vector
            # [1, embed_dim, G*G] -> [1, embed_dim, 1] -> [1, embed_dim]
            pooled = self.global_pool(out.transpose(1, 2)).squeeze(-1)
            out_features.append(pooled)
            
        return torch.cat(out_features, dim=0) # [B, embed_dim]
