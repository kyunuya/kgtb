import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class GraphEncoder(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()

        self.node_emb = nn.Embedding(num_nodes, hidden_dim, max_norm=1.0)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=num_relations)
            )

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.node_emb.weight

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type)

            if i < len(self.layers) - 1:
                x = F.relu(x)

        return x