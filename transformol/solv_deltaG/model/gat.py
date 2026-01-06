import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    """GAT-based model using Graph Attention Network."""
    
    def __init__(self, node_feature_dim, hidden_dim=64, num_layers=3, heads=4):
        super(GATModel, self).__init__()
        
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.2))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.2))
        
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.2))
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.node_encoder(x)
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation after last layer
                x = F.elu(x)
        
        # Separate pooling for solute and solvent
        solute_mask = (batch == 0)
        solvent_mask = (batch == 1)
        
        # Handle case where mask might be empty
        if solute_mask.sum() > 0:
            solute_repr = x[solute_mask].mean(dim=0, keepdim=True)
        else:
            solute_repr = t.zeros(1, x.size(1), device=x.device)
        
        if solvent_mask.sum() > 0:
            solvent_repr = x[solvent_mask].mean(dim=0, keepdim=True)
        else:
            solvent_repr = t.zeros(1, x.size(1), device=x.device)
        
        combined = t.cat([solute_repr, solvent_repr], dim=1)
        
        return self.mlp(combined)
    