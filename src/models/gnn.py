"""CropOSGNN — heterogeneous GNN for Thai farm-level precipitation forecasting."""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class CropOSGNN(nn.Module):
    """
    Heterogeneous message-passing GNN.

    Input:  HeteroData graph — era5 grid nodes + metar airport nodes + farm target nodes
    Output: (n_farms, n_horizons) precipitation probabilities via sigmoid

    Architecture:
      - Input projection: each node type → hidden_dim
      - N × HeteroSAGE layers: era5→metar, era5→farm, metar→farm
      - Farm node MLP head → sigmoid → probabilities
    """

    def __init__(
        self,
        era5_in: int,
        metar_in: int,
        hidden: int,
        n_horizons: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.era5_proj = nn.Linear(era5_in, hidden)
        self.metar_proj = nn.Linear(metar_in, hidden)
        self.farm_proj = nn.Linear(1, hidden)
        self.drop = nn.Dropout(dropout)
        # Hidden layers propagate all edge types (era5→metar feeds later metar→farm).
        # The final layer omits era5→metar so every weight lies on the gradient path
        # through farm → head → loss.
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            edge_types: dict = {
                ("era5", "to", "farm"): SAGEConv(hidden, hidden),
                ("metar", "to", "farm"): SAGEConv(hidden, hidden),
            }
            if i < num_layers - 1:
                edge_types[("era5", "to", "metar")] = SAGEConv(hidden, hidden)
            self.convs.append(HeteroConv(edge_types, aggr="sum"))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_horizons), nn.Sigmoid(),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = {
            "era5": self.era5_proj(data["era5"].x),
            "metar": self.metar_proj(data["metar"].x),
            "farm": self.farm_proj(data["farm"].x),
        }
        edge_index_dict = {
            k: v.edge_index
            for k, v in data.edge_items()
            if hasattr(v, "edge_index")
        }
        for conv in self.convs:
            new_x = conv(x, edge_index_dict)
            # Preserve embeddings for source-only node types (e.g. era5) that
            # HeteroConv omits from its output because they are never a destination.
            for node_type in x:
                if node_type in new_x:
                    x[node_type] = torch.relu(self.drop(new_x[node_type]))
                # else: keep previous embedding unchanged
        return self.head(x["farm"])  # (n_farms, n_horizons)
