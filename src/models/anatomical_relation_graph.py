import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomicalRelationGraph(nn.Module):
    """
    Part E: Anatomical Relation Graph (ARG)
    Constructs a sequential graph of L1-L5 vertebrae using the morphology embeddings
    and explicit geometric relationships as edge weights/features.
    Produces refined vertebral embeddings and a global spine embedding.
    """
    def __init__(self, node_dim=256, edge_dim=4, hidden_dim=256):
        super().__init__()
        # GAT-style node update incorporating edge features
        self.node_update = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, node_dim)
        )
        # Global spine embedding pooling
        self.global_pool = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(self, vme_embeddings, boxes, labels):
        """
        Processes a single image's vertebrae (since graphs vary per image).
        
        Args:
            vme_embeddings: Tensor [N, node_dim] from VME
            boxes: Tensor [N, 4]
            labels: Tensor [N] containing int labels (1=L1, 2=L2, etc.)
            
        Returns:
            refined_nodes: Tensor [N, node_dim]
            global_spine: Tensor [node_dim]
        """
        N = vme_embeddings.size(0)
        if N == 0:
            return vme_embeddings, torch.zeros(vme_embeddings.size(1), device=vme_embeddings.device)
            
        # We enforce sequential connections based on the labels
        # e.g., L1 connected to L2, L2 to L3.
        # Find adjacent pairs
        edges = []
        edge_feats = []
        
        # Sort by label to cleanly build sequential edges
        sorted_indices = torch.argsort(labels)
        sorted_labels = labels[sorted_indices]
        
        for i in range(len(sorted_labels) - 1):
            idx_u = sorted_indices[i]
            idx_v = sorted_indices[i+1]
            
            lbl_u = sorted_labels[i].item()
            lbl_v = sorted_labels[i+1].item()
            
            # Connect only anatomically adjacent vertebrae (e.g., L1-L2, L4-L5)
            if lbl_v == lbl_u + 1:
                edges.append((idx_u, idx_v))
                edges.append((idx_v, idx_u)) # Bi-directional
                
                # Compute relative geometric features
                box_u = boxes[idx_u]
                box_v = boxes[idx_v]
                
                h_u = box_u[3] - box_u[1]
                w_u = box_u[2] - box_u[0]
                
                h_v = box_v[3] - box_v[1]
                w_v = box_v[2] - box_v[0]
                
                center_y_u = box_u[1] + h_u / 2
                center_y_v = box_v[1] + h_v / 2
                
                # 4 edge features: relative height, relative width, vertical distance, overlap
                rel_h = h_v / (h_u + 1e-6)
                rel_w = w_v / (w_u + 1e-6)
                vert_dist = (center_y_v - center_y_u) / (h_u + 1e-6)
                # Simple IoU logic can be skipped for now in favor of center distances
                
                feat = torch.stack([rel_h, rel_w, vert_dist, vert_dist**2])
                edge_feats.append(feat)
                edge_feats.append(feat) # Same for reverse edge for simplicity
                
        # Message passing
        refined_nodes = vme_embeddings.clone()
        if len(edges) > 0:
            for i, (u, v) in enumerate(edges):
                msg_input = torch.cat([vme_embeddings[u], vme_embeddings[v], edge_feats[i]], dim=0)
                msg = self.node_update(msg_input)
                # Aggregate (residual connection)
                refined_nodes[v] = refined_nodes[v] + msg
                
        # Global pooling (e.g., mean pool followed by MLP)
        global_repr = refined_nodes.mean(dim=0)
        global_spine = self.global_pool(global_repr) + global_repr
        
        return refined_nodes, global_spine
