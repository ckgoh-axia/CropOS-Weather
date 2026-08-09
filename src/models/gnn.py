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
      era5  — coarse atmospheric grid (ERA5-Land, 7 features)
      metar — METAR airport weather stations (actual observations at inference time).
              Features: [precip_mm, rain_event, tmpf, dwpf, relh, drct, sknt, alti, vsby]
              Missing observations pad with 0. Inductive: new stations need no retraining.
      farm  — target prediction node (placeholder feature = 0)

    DropNode (metar_dropout):
      During training, each metar node is independently zeroed with probability
      metar_dropout. This forces the model to learn from ERA5 alone on ~40% of
      examples, making it robust when no nearby station exists AND inductive to new
      stations added at inference without retraining.

    Output: (n_farms, n_horizons) precipitation probabilities in [0, 1].
    """

    METAR_FEATURES = [
        "precip_mm", "rain_event",
        "tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby",
    ]

    def __init__(
        self,
        era5_in: int,
        hidden: int,
        n_horizons: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        metar_dropout: float = 0.4,
        metar_in: int | None = None,
    ):
        super().__init__()
        # metar_in defaults to the 9-feature real-time METAR observation set.
        # Pass metar_in=<N> when using a different feature set (e.g. extended).
        if metar_in is None:
            metar_in = len(self.METAR_FEATURES)
        self.metar_dropout = metar_dropout

        self.era5_proj = nn.Linear(era5_in, hidden)
        self.metar_proj = nn.Linear(metar_in, hidden)
        self.farm_proj = nn.Linear(1, hidden)
        self.drop = nn.Dropout(dropout)

        # Hidden layers: all edge types so era5→metar feeds later metar→farm
        # Final layer: only paths that reach farm → head → loss (no dead weights)
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
        x_metar = data["metar"].x

        # DropNode: randomly zero entire metar nodes during training.
        # The model learns ERA5-only predictions ~40% of the time, making it
        # inductive: new stations added at inference improve predictions without retraining.
        if self.training and self.metar_dropout > 0:
            keep_prob = 1.0 - self.metar_dropout
            mask = torch.bernoulli(
                torch.full((x_metar.shape[0], 1), keep_prob, device=x_metar.device)
            )
            x_metar = x_metar * mask

        x = {
            "era5": self.era5_proj(data["era5"].x),
            "metar": self.metar_proj(x_metar),
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
                # else: preserve embedding for source-only nodes (era5, metar in final layer)

        return self.head(x["farm"])  # (n_farms, n_horizons)
