import torch
import torch.nn as nn

class MessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(MessagePassingLayer, self).__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(self, nodes, edge_indices, edge_features):
        """
        nodes: (N, node_dim)
        edge_indices: (2, E)
        edge_features: (E, edge_dim)
        """
        N = nodes.shape[0]
        E = edge_indices.shape[1]
        
        if N == 0 or E == 0:
            return nodes
            
        src, dst = edge_indices
        
        # Gather node features for edges
        src_nodes = nodes[src] # (E, node_dim)
        dst_nodes = nodes[dst] # (E, node_dim)
        
        # Compute messages
        msg_input = torch.cat([src_nodes, dst_nodes, edge_features], dim=1) # (E, 2*node_dim + edge_dim)
        messages = self.message_mlp(msg_input) # (E, hidden_dim)
        
        # Aggregate messages (sum pooling)
        aggregated = torch.zeros(N, messages.shape[1], device=nodes.device)
        aggregated.index_add_(0, dst, messages)
        
        # Update node features
        update_input = torch.cat([nodes, aggregated], dim=1) # (N, node_dim + hidden_dim)
        updated_nodes = nodes + self.update_mlp(update_input) # Residual connection
        
        return updated_nodes

class SpatialGraphEncoder(nn.Module):
    def __init__(self, node_dim=6, edge_dim=5, embed_dim=64):
        super(SpatialGraphEncoder, self).__init__()
        
        self.node_emb = nn.Sequential(
            nn.Linear(node_dim, 32),
            nn.ReLU()
        )
        
        self.layer1 = MessagePassingLayer(32, edge_dim, 32)
        self.layer2 = MessagePassingLayer(32, edge_dim, 32)
        
        self.out_emb = nn.Sequential(
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )
        
    def forward(self, graph_dict):
        nodes = graph_dict['nodes'] # (N, 6)
        edge_indices = graph_dict['edge_indices'] # (2, E)
        edge_features = graph_dict['edge_features'] # (E, 5)
        
        if nodes.shape[0] == 0:
            # Handle empty graph
            return torch.zeros(1, self.out_emb[0].out_features, device=nodes.device)
            
        x = self.node_emb(nodes)
        
        x = self.layer1(x, edge_indices, edge_features)
        x = torch.relu(x)
        
        x = self.layer2(x, edge_indices, edge_features)
        x = torch.relu(x)
        
        x = self.out_emb(x)
        
        # Global mean pooling
        graph_embed = torch.mean(x, dim=0, keepdim=True) # (1, embed_dim)
        return graph_embed
