# Layers for GNN

import torch as t
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MPNNConv(MessagePassing):
    """Optimized Message Passing Neural Network layer for faster training."""
    
    def __init__(self, in_channels, out_channels, edge_dim):
        super(MPNNConv, self).__init__(aggr='add')
        # Simplified single-layer MLPs for speed
        self.message_lin = nn.Linear(in_channels + edge_dim, out_channels)
        self.update_lin = nn.Linear(in_channels + out_channels, out_channels)
        self.activation = nn.ReLU(inplace=True)  # In-place for memory efficiency
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # Single linear transformation for speed
        tmp = t.cat([x_j, edge_attr], dim=1)
        return self.activation(self.message_lin(tmp))
    
    def update(self, aggr_out, x):
        # Single linear transformation for speed
        tmp = t.cat([x, aggr_out], dim=1)
        return self.activation(self.update_lin(tmp))
    

class GCNConv(MessagePassing):
    """Custom Graph Convolutional Network layer (for reference)."""
    
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = self.add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Linear transformation
        x = self.lin(x)
        
        # Normalize
        row, col = edge_index
        deg = t.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def add_self_loops(self, edge_index, num_nodes):
        loop_index = t.arange(0, num_nodes, dtype=t.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = t.cat([edge_index, loop_index], dim=1)
        return edge_index, None
    