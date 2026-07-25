"""CropOSGNN — heterogeneous GNN for Thai farm-level precipitation forecasting."""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class CropOSGNN(nn.Module):
    """
    Heterogeneous message-passing GNN.

    Node types:
      era5          — coarse atmospheric grid (ERA5-Land, 7 features)
      local_station — any surface weather station: airport METAR, farm sensor, etc.
                      Always 5 features: [precip_mm, temperature, humidity, windspeed, pressure]
                      Missing sensors pad with 0. Inductive: new stations need no retraining.
      farm          — target prediction node (placeholder feature = 0)

    DropNode (local_station_dropout):
      During training, each local_station node is independently zeroed with probability
      local_station_dropout. This forces the model to learn from ERA5 alone on ~40% of
      examples, making it robust when no nearby station exists AND inductive to new
      stations added at inference without retraining.

    Output: (n_farms, n_horizons) precipitation probabilities in [0, 1].
    """

    LOCAL_STATION_FEATURES = ["precip_mm", "temperature", "humidity", "windspeed", "pressure"]

    def __init__(
        self,
        era5_in: int,
        hidden: int,
        n_horizons: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        local_station_dropout: float = 0.4,
    ):
        super().__init__()
        local_station_in = len(self.LOCAL_STATION_FEATURES)  # always 5
        self.local_station_dropout = local_station_dropout

        self.era5_proj = nn.Linear(era5_in, hidden)
        self.local_station_proj = nn.Linear(local_station_in, hidden)
        self.farm_proj = nn.Linear(1, hidden)
        self.drop = nn.Dropout(dropout)

        # Hidden layers: all edge types so era5→local_station feeds later local_station→farm
        # Final layer: only paths that reach farm → head → loss (no dead weights)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            edge_types: dict = {
                ("era5", "to", "farm"): SAGEConv(hidden, hidden),
                ("local_station", "to", "farm"): SAGEConv(hidden, hidden),
            }
            if i < num_layers - 1:
                edge_types[("era5", "to", "local_station")] = SAGEConv(hidden, hidden)
            self.convs.append(HeteroConv(edge_types, aggr="sum"))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_horizons), nn.Sigmoid(),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_local = data["local_station"].x

        # DropNode: randomly zero entire local_station nodes during training.
        # The model learns ERA5-only predictions ~40% of the time, making it
        # inductive: new stations added at inference improve predictions without retraining.
        if self.training and self.local_station_dropout > 0:
            keep_prob = 1.0 - self.local_station_dropout
            mask = torch.bernoulli(
                torch.full((x_local.shape[0], 1), keep_prob, device=x_local.device)
            )
            x_local = x_local * mask

        x = {
            "era5": self.era5_proj(data["era5"].x),
            "local_station": self.local_station_proj(x_local),
            "farm": self.farm_proj(data["farm"].x),
        }
        edge_index_dict = {
            k: v.edge_index
            for k, v in data.edge_items()
            if hasattr(v, "edge_index")
        }
        for conv in self.convs:
            new_x = conv(x, edge_index_dict)
            for node_type in x:
                if node_type in new_x:
                    x[node_type] = torch.relu(self.drop(new_x[node_type]))
                # else: preserve embedding for source-only nodes (era5)

        return self.head(x["farm"])  # (n_farms, n_horizons)
