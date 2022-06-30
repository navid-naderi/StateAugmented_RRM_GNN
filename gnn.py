import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LEConv
from torch_scatter import scatter

# backbone GNN class
class gnn_backbone(torch.nn.Module):
    def __init__(self, num_features_list):
        super(gnn_backbone, self).__init__()
        num_layers = len(num_features_list)
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(LEConv(num_features_list[i], num_features_list[i + 1]))
            
    def forward(self, y, edge_index, edge_weight):
        for i, layer in enumerate(self.layers):
            y = layer(y, edge_index=edge_index, edge_weight=edge_weight)
            y = F.leaky_relu(y)
            
        return y
    
# main GNN module
class GNN(torch.nn.Module):
    def __init__(self, num_features_list, P_max):
        super(GNN, self).__init__()
        self.gnn_backbone = gnn_backbone(num_features_list)
        self.b_p = nn.Linear(num_features_list[-1], 1, bias=False)
        self.P_max = P_max
        
    def forward(self, y, edge_index, edge_weight, transmitters_index):
        y = self.gnn_backbone(y, edge_index, edge_weight) # derive node embeddings
        Tx_embeddings = scatter(y, transmitters_index, dim=0, reduce='mean')        
        p = self.P_max * torch.sigmoid(self.b_p(Tx_embeddings)) # derive power levels for transmitters
        return p