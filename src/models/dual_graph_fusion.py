import torch
import torch.nn as nn

class DualGraphFusion(nn.Module):
    """
    Part G: Dual-Graph Fusion
    Fuses the Anatomical Relation Graph (ARG) and the Pathology Relation Graph.
    Uses Cross-Attention where Pathology nodes query the Anatomical nodes,
    imbuing the pathology regions with explicit anatomical context.
    """
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        
        # Pathology (Query) attends to Anatomy (Key, Value)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, arg_nodes_list, path_graph_batch):
        """
        Args:
            arg_nodes_list: List of Tensors [N_i, embed_dim] for B images.
            path_graph_batch: Tensor [B, G*G, embed_dim]
            
        Returns:
            fused_pathology: Tensor [B, G*G, embed_dim]
        """
        B = path_graph_batch.size(0)
        fused_pathology = []
        
        # Since ARG nodes vary per image (N_i vertebrae), process item by item
        for i in range(B):
            path_q = path_graph_batch[i:i+1] # [1, G*G, embed_dim]
            arg_kv = arg_nodes_list[i].unsqueeze(0) # [1, N_i, embed_dim]
            
            if arg_kv.size(1) == 0:
                # No anatomy localized, fallback to identity mapping
                fused_pathology.append(path_q)
                continue
                
            attn_out, _ = self.cross_attn(query=path_q, key=arg_kv, value=arg_kv)
            
            # Add & Norm
            out1 = self.norm1(path_q + attn_out)
            
            # FFN
            out2 = self.ffn(out1)
            out = self.norm2(out1 + out2)
            
            fused_pathology.append(out)
            
        return torch.cat(fused_pathology, dim=0) # [B, G*G, embed_dim]
