# tests/unit/test_graph_builder.py
import torch

from src.features.graph_builder import build_heterogeneous_graph, haversine_km


def _era5():
    return [{"lat": 15.0, "lon": 102.0, "feats": [1.0] * 7}]


def _station_nearby():
    return [{"lat": 15.25, "lon": 102.5, "feats": [0.5] * 5}]


def _station_far():
    return [{"lat": 1.0, "lon": 104.0, "feats": [0.5] * 5}]  # ~1600 km away


def _farm():
    return [{"lat": 15.1, "lon": 102.3, "farm_id": "f1"}]


def test_haversine_bangkok_chiangmai():
    dist = haversine_km(13.75, 100.50, 18.77, 98.96)
    assert 560 < dist < 620


def test_haversine_same_point_is_zero():
    assert haversine_km(15.0, 102.0, 15.0, 102.0) == 0.0


def test_graph_has_all_three_node_types():
    graph = build_heterogeneous_graph(
        era5_nodes=_era5(),
        local_station_nodes=_station_nearby(),
        farm_nodes=_farm(),
        edge_radius_km=200,
    )
    assert "era5" in graph.node_types
    assert "local_station" in graph.node_types
    assert "farm" in graph.node_types


def test_graph_node_feature_shapes():
    graph = build_heterogeneous_graph(
        era5_nodes=_era5(),
        local_station_nodes=_station_nearby(),
        farm_nodes=_farm(),
        edge_radius_km=200,
    )
    assert graph["era5"].x.shape == (1, 7)
    assert graph["local_station"].x.shape == (1, 5)
    assert graph["farm"].x.shape == (1, 1)


def test_graph_edges_connect_nearby_nodes():
    graph = build_heterogeneous_graph(
        era5_nodes=_era5(),
        local_station_nodes=_station_nearby(),
        farm_nodes=_farm(),
        edge_radius_km=200,
    )
    assert graph["era5", "to", "local_station"].edge_index.shape[1] > 0
    assert graph["local_station", "to", "farm"].edge_index.shape[1] > 0
    assert graph["era5", "to", "farm"].edge_index.shape[1] > 0


def test_graph_edges_exclude_distant_nodes():
    graph = build_heterogeneous_graph(
        era5_nodes=_era5(),
        local_station_nodes=_station_far(),
        farm_nodes=_farm(),
        edge_radius_km=100,
    )
    assert graph["era5", "to", "local_station"].edge_index.shape[1] == 0
    assert graph["local_station", "to", "farm"].edge_index.shape[1] == 0


def test_graph_no_local_stations():
    """ERA5-only graph: local_station tensor exists but is empty."""
    graph = build_heterogeneous_graph(
        era5_nodes=_era5(),
        local_station_nodes=[],
        farm_nodes=_farm(),
        edge_radius_km=200,
    )
    assert graph["local_station"].x.shape[0] == 0
    assert graph["era5", "to", "local_station"].edge_index.shape[1] == 0


def test_graph_multiple_farms():
    farms = [
        {"lat": 15.1, "lon": 102.3, "farm_id": "f1"},
        {"lat": 15.5, "lon": 102.1, "farm_id": "f2"},
        {"lat": 14.9, "lon": 101.8, "farm_id": "f3"},
    ]
    graph = build_heterogeneous_graph(
        era5_nodes=_era5(),
        local_station_nodes=_station_nearby(),
        farm_nodes=farms,
        edge_radius_km=300,
    )
    assert graph["farm"].x.shape[0] == 3
