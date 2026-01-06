import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .layer import MPNNConv

class MPNNModel(nn.Module):
    """Optimized MPNN-based model for faster solvation free energy prediction."""
    
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=48, num_layers=2):
        super(MPNNModel, self).__init__()
        
        # Smaller hidden dimension for speed
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        # Fewer layers for faster training
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(MPNNConv(hidden_dim, hidden_dim, edge_feature_dim))
        
        # Simplified output MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for solute + solvent pooling
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_encoder(x)
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Separate pooling for solute and solvent
        solute_mask = (batch == 0)
        solvent_mask = (batch == 1)
        
        solute_repr = x[solute_mask].mean(dim=0, keepdim=True)
        solvent_repr = x[solvent_mask].mean(dim=0, keepdim=True)
        
        # Combine representations
        combined = t.cat([solute_repr, solvent_repr], dim=1)
        
        return self.mlp(combined)
    