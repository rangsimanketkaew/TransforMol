import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    """GCN-based model using PyTorch Geometric's GCNConv layer."""
    
    def __init__(self, node_feature_dim, hidden_dim=64, num_layers=3):
        super(GCNModel, self).__init__()
        
        # Input layer
        # self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.node_encoder = GCNConv(node_feature_dim, hidden_dim)
        
        # Hidden layer
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        """Construct a network"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # x = self.node_encoder(x)
        x = self.node_encoder(x, edge_index)
        
        # for conv in self.convs:
        #     x = conv(x, edge_index)
        #     x = F.relu(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # Don't apply activation after last layer
            if i < len(self.convs) - 1:  
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
    