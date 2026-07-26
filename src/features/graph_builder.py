"""Build heterogeneous PyTorch Geometric graph for CropOS."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
from torch_geometric.data import HeteroData


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _edges_within_radius(
    src_nodes: List[Dict], dst_nodes: List[Dict], radius_km: float
) -> torch.Tensor:
    src_idx, dst_idx = [], []
    for i, s in enumerate(src_nodes):
        for j, d in enumerate(dst_nodes):
            if haversine_km(s["lat"], s["lon"], d["lat"], d["lon"]) <= radius_km:
                src_idx.append(i)
                dst_idx.append(j)
    if not src_idx:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src_idx, dst_idx], dtype=torch.long)


def build_heterogeneous_graph(
    era5_nodes: List[Dict[str, Any]],
    local_station_nodes: List[Dict[str, Any]],
    farm_nodes: List[Dict[str, Any]],
    edge_radius_km: float = 200.0,
) -> HeteroData:
    """Build PyG HeteroData with era5, local_station, and farm node types.

    local_station nodes are agnostic to source — Phase 1 uses airport METAR
    stations; Phase 2 adds cheap IoT farm sensors as the same node type without
    retraining (SAGEConv weights are shared, not per-node-identity).
    """
    data = HeteroData()
    data["era5"].x = torch.tensor(
        [n["feats"] for n in era5_nodes], dtype=torch.float
    )
    data["local_station"].x = torch.tensor(
        [n["feats"] for n in local_station_nodes], dtype=torch.float
    )
    data["farm"].x = torch.zeros((len(farm_nodes), 1), dtype=torch.float)
    data["era5", "to", "local_station"].edge_index = _edges_within_radius(
        era5_nodes, local_station_nodes, edge_radius_km
    )
    data["era5", "to", "farm"].edge_index = _edges_within_radius(
        era5_nodes, farm_nodes, edge_radius_km
    )
    data["local_station", "to", "farm"].edge_index = _edges_within_radius(
        local_station_nodes, farm_nodes, edge_radius_km
    )
    return data
