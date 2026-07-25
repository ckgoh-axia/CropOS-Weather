# tests/unit/test_graph_builder.py
import pytest
import torch
from src.features.graph_builder import build_heterogeneous_graph, haversine_km


def test_haversine_bangkok_chiangmai():
    # Bangkok (13.75, 100.50) to Chiang Mai (18.77, 98.96) ≈ 582 km (great-circle)
    dist = haversine_km(13.75, 100.50, 18.77, 98.96)
    assert 560 < dist < 620


def test_graph_has_all_three_node_types():
    graph = build_heterogeneous_graph(
        era5_nodes=[{"lat": 15.0, "lon": 102.0, "feats": [1.0] * 7}],
        local_station_nodes=[{"lat": 15.25, "lon": 102.5, "feats": [0.5] * 5}],
        farm_nodes=[{"lat": 15.1, "lon": 102.3, "farm_id": "f1"}],
        edge_radius_km=200,
    )
    assert "era5" in graph.node_types
    assert "local_station" in graph.node_types
    assert "farm" in graph.node_types


def test_graph_edges_connect_nearby_nodes():
    graph = build_heterogeneous_graph(
        era5_nodes=[{"lat": 15.0, "lon": 102.0, "feats": [1.0] * 7}],
        local_station_nodes=[{"lat": 15.25, "lon": 102.5, "feats": [0.5] * 5}],
        farm_nodes=[{"lat": 15.1, "lon": 102.3, "farm_id": "f1"}],
        edge_radius_km=200,
    )
    # All three nodes are within 200km of each other — expect edges
    assert graph["era5", "to", "local_station"].edge_index.shape[1] > 0
    assert graph["local_station", "to", "farm"].edge_index.shape[1] > 0


def test_graph_edges_exclude_distant_nodes():
    graph = build_heterogeneous_graph(
        era5_nodes=[{"lat": 15.0, "lon": 102.0, "feats": [1.0] * 7}],
        local_station_nodes=[{"lat": 1.0, "lon": 104.0, "feats": [0.5] * 5}],  # Singapore area
        farm_nodes=[{"lat": 15.1, "lon": 102.3, "farm_id": "f1"}],
        edge_radius_km=100,
    )
    # ~1600km away — no edge within 100km radius
    assert graph["era5", "to", "local_station"].edge_index.shape[1] == 0


def test_graph_no_local_stations():
    """Graph must be constructable with zero local_station nodes (ERA5-only mode)."""
    graph = build_heterogeneous_graph(
        era5_nodes=[{"lat": 15.0, "lon": 102.0, "feats": [1.0] * 7}],
        local_station_nodes=[],
        farm_nodes=[{"lat": 15.1, "lon": 102.3, "farm_id": "f1"}],
        edge_radius_km=200,
    )
    assert graph["local_station"].x.shape[0] == 0
    assert graph["era5", "to", "local_station"].edge_index.shape[1] == 0
