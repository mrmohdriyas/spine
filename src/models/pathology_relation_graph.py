import torch
import torch.nn as nn

class PathologyRelationGraph(nn.Module):
    """
    Part F: Pathology Relation Graph
    Generates pathology candidate nodes from image features (e.g., a spatial grid),
    predicts their pathology probabilities, and connects them spatially.
    Ground-truth pathology boxes are ONLY used to supervise the `region_classifier` during training,
    not fed as inputs to inference.
    """
    def __init__(self, in_channels=256, node_dim=256, num_pathologies=8, grid_size=7):
        super().__init__()
        self.grid_size = grid_size
        self.node_dim = node_dim
        
        # Maps backbone spatial features to node embeddings
        self.feature_proj = nn.Conv2d(in_channels, node_dim, kernel_size=1)
        
        # Predicts pathology probability for each region node
        self.region_classifier = nn.Linear(node_dim, num_pathologies)
        
        # Graph reasoning layer (self-attention over spatial nodes)
        self.gnn = nn.MultiheadAttention(embed_dim=node_dim, num_heads=4, batch_first=True)
        
    def forward(self, feature_map):
        """
        Args:
            feature_map: Tensor [B, C, H, W] from the backbone.
            
        Returns:
            pathology_graph_features: Tensor [B, grid*grid, node_dim]
            region_pathology_logits: Tensor [B, grid*grid, num_pathologies]
        """
        B, C, H, W = feature_map.shape
        
        # Adaptive pool to fixed grid if necessary
        pooled_features = nn.functional.adaptive_avg_pool2d(feature_map, (self.grid_size, self.grid_size))
        
        # Project to node dim
        node_features = self.feature_proj(pooled_features) # [B, node_dim, G, G]
        
        # Flatten spatial dimensions to create a sequence of nodes
        nodes = node_features.view(B, self.node_dim, -1).transpose(1, 2) # [B, G*G, node_dim]
        
        # Predict pathology logits for each candidate node
        region_logits = self.region_classifier(nodes) # [B, G*G, 8]
        
        # Pathological Graph Reasoning (Fully connected attention over regions)
        # Spatial relationships can be explicitly added via positional embeddings,
        # but standard self-attention allows dense connection weighted by visual+pathology context.
        refined_nodes, _ = self.gnn(nodes, nodes, nodes)
        
        # Residual connection
        pathology_graph_features = nodes + refined_nodes
        
        return pathology_graph_features, region_logits
