"""Build heterogeneous PyTorch Geometric graph for CropOS.

Node types: era5, metar, farm.

Edge construction strategy:
- era5 -> metar : k-NN bipartite — each METAR station connects to its
  ``era5_to_metar_k`` nearest ERA5 grid points, guaranteeing every station
  receives exactly k global-context inputs regardless of grid density.
- era5 -> farm  : radius-based (``edge_radius_km``).
- metar -> farm : radius-based (``edge_radius_km``).
- metar -> metar: symmetric k-NN with configurable k (``metar_to_metar_k``).

All node types store a ``.pos`` tensor (lat/lon degrees) for use in
relative-position message passing inside the GNN.
"""
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


def _knn_edges(nodes: List[Dict], k: int = 4) -> torch.Tensor:
    """Connect each node to its k nearest neighbours (bidirectional, deduped)."""
    n = len(nodes)
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    edge_set: set[tuple[int, int]] = set()
    for i, src in enumerate(nodes):
        dists = sorted(
            (haversine_km(src["lat"], src["lon"], nodes[j]["lat"], nodes[j]["lon"]), j)
            for j in range(n) if j != i
        )
        for _, j in dists[:k]:
            edge_set.add((i, j))
            edge_set.add((j, i))
    if not edge_set:
        return torch.zeros((2, 0), dtype=torch.long)
    edges = sorted(edge_set)
    return torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)


def _knn_edges_bipartite(
    src_nodes: List[Dict], dst_nodes: List[Dict], k: int = 8
) -> torch.Tensor:
    """Connect each dst node to its k nearest src nodes (unidirectional: src->dst).

    Used for ERA5->metar edges: each METAR station connects to its 8 nearest
    ERA5 grid points. Ensures every station gets exactly k global-context inputs
    regardless of grid density, unlike radius-based filtering.

    Returns:
        edge_index (2, num_edges) where [0] = src indices, [1] = dst indices.
    """
    src_idx, dst_idx = [], []
    for j, d in enumerate(dst_nodes):
        dists = sorted(
            (haversine_km(s["lat"], s["lon"], d["lat"], d["lon"]), i)
            for i, s in enumerate(src_nodes)
        )
        for _, i in dists[:min(k, len(src_nodes))]:
            src_idx.append(i)
            dst_idx.append(j)
    if not src_idx:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src_idx, dst_idx], dtype=torch.long)


def build_heterogeneous_graph(
    era5_nodes: List[Dict[str, Any]],
    metar_nodes: List[Dict[str, Any]],
    farm_nodes: List[Dict[str, Any]],
    edge_radius_km: float = 200.0,
    era5_to_metar_k: int = 8,
    metar_to_metar_k: int = 4,
) -> HeteroData:
    """Build PyG HeteroData with era5, metar, and farm node types.

    Edge construction:
    - era5 -> metar : k-NN bipartite using ``era5_to_metar_k``; each METAR
      station is connected to exactly k nearest ERA5 grid points, providing
      uniform global-context coverage independent of local grid density.
    - era5 -> farm  : radius-based within ``edge_radius_km`` km.
    - metar -> farm : radius-based within ``edge_radius_km`` km.
    - metar -> metar: symmetric k-NN with ``metar_to_metar_k`` neighbours.

    All node types expose a ``.pos`` tensor of shape (N, 2) holding raw
    lat/lon degrees. The GNN can use these to compute relative-position
    encodings inside message-passing layers.

    Args:
        era5_nodes:       List of dicts with keys ``lat``, ``lon``, ``feats``.
        metar_nodes:      List of dicts with keys ``lat``, ``lon``, ``feats``.
        farm_nodes:       List of dicts with keys ``lat``, ``lon``.
        edge_radius_km:   Radius used for era5->farm and metar->farm edges.
        era5_to_metar_k:  Number of nearest ERA5 grid points per METAR station.
        metar_to_metar_k: Number of nearest METAR neighbours per station.

    Returns:
        A ``HeteroData`` object ready for PyG message passing.
    """
    data = HeteroData()

    # Node features
    data["era5"].x = torch.tensor(
        [n["feats"] for n in era5_nodes], dtype=torch.float
    )
    data["metar"].x = torch.tensor(
        [n["feats"] for n in metar_nodes], dtype=torch.float
    )
    data["farm"].x = torch.zeros((len(farm_nodes), 1), dtype=torch.float)

    # Edges
    data["era5", "to", "metar"].edge_index = _knn_edges_bipartite(
        era5_nodes, metar_nodes, era5_to_metar_k
    )
    data["era5", "to", "farm"].edge_index = _edges_within_radius(
        era5_nodes, farm_nodes, edge_radius_km
    )
    data["metar", "to", "farm"].edge_index = _edges_within_radius(
        metar_nodes, farm_nodes, edge_radius_km
    )
    data["metar", "to", "metar"].edge_index = _knn_edges(metar_nodes, k=metar_to_metar_k)

    # Node positions — raw lat/lon degrees, used for relative position in messages
    data["era5"].pos = torch.tensor(
        [[n["lat"], n["lon"]] for n in era5_nodes], dtype=torch.float
    )
    data["metar"].pos = torch.tensor(
        [[n["lat"], n["lon"]] for n in metar_nodes], dtype=torch.float
    )
    data["farm"].pos = torch.tensor(
        [[n["lat"], n["lon"]] for n in farm_nodes], dtype=torch.float
    )

    return data
