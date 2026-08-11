# tests/unit/test_graph_builder.py
import torch

from src.features.graph_builder import build_heterogeneous_graph, haversine_km


def test_haversine_bangkok_chiangmai():
    # Bangkok (13.75, 100.50) to Chiang Mai (18.77, 98.96) ≈ 582 km (great-circle)
    dist = haversine_km(13.75, 100.50, 18.77, 98.96)
    assert 560 < dist < 620


def test_graph_has_all_three_node_types():
    graph = build_heterogeneous_graph(
        era5_nodes=[{"lat": 15.0, "lon": 102.0, "feats": [1.0] * 7}],
        metar_nodes=[{"lat": 15.25, "lon": 102.5, "feats": [0.5] * 5}],
        farm_nodes=[{"lat": 15.1, "lon": 102.3, "farm_id": "f1"}],
        edge_radius_km=200,
    )
    assert "era5" in graph.node_types
    assert "metar" in graph.node_types
    assert "farm" in graph.node_types


def test_graph_edges_connect_nearby_nodes():
    graph = build_heterogeneous_graph(
        era5_nodes=[{"lat": 15.0, "lon": 102.0, "feats": [1.0] * 7}],
        metar_nodes=[{"lat": 15.25, "lon": 102.5, "feats": [0.5] * 5}],
        farm_nodes=[{"lat": 15.1, "lon": 102.3, "farm_id": "f1"}],
        edge_radius_km=200,
    )
    # All three nodes are within 200km of each other — expect edges
    assert graph["era5", "to", "metar"].edge_index.shape[1] > 0
    assert graph["metar", "to", "farm"].edge_index.shape[1] > 0


def test_era5_to_metar_uses_knn_not_radius():
    """ERA5→metar edges use bipartite k-NN: every METAR station always gets
    exactly min(k, n_era5) nearest ERA5 nodes regardless of distance."""
    # METAR station in Singapore area (~1600 km from ERA5 node in northern Thailand).
    # With radius-based filtering this would produce 0 edges, but k-NN always connects.
    graph = build_heterogeneous_graph(
        era5_nodes=[{"lat": 15.0, "lon": 102.0, "feats": [1.0] * 7}],
        metar_nodes=[{"lat": 1.0, "lon": 104.0, "feats": [0.5] * 5}],
        farm_nodes=[{"lat": 15.1, "lon": 102.3, "farm_id": "f1"}],
        edge_radius_km=100,
        era5_to_metar_k=8,
    )
    # k-NN guarantees at least 1 edge even across 1600 km (min(k, n_era5) = min(8,1) = 1)
    assert graph["era5", "to", "metar"].edge_index.shape[1] == 1


def test_era5_to_metar_knn_exact_count():
    """Each METAR station connects to exactly min(k, n_era5) ERA5 nodes."""
    n_era5 = 10
    n_metar = 3
    k = 4
    era5_nodes = [{"lat": 13.0 + i * 0.5, "lon": 100.0 + i * 0.5, "feats": [1.0] * 7}
                  for i in range(n_era5)]
    metar_nodes = [{"lat": 13.2 + i * 0.3, "lon": 100.2 + i * 0.3, "feats": [0.5] * 5}
                   for i in range(n_metar)]
    farm_nodes = [{"lat": 13.1, "lon": 100.1, "farm_id": "f1"}]

    graph = build_heterogeneous_graph(
        era5_nodes=era5_nodes,
        metar_nodes=metar_nodes,
        farm_nodes=farm_nodes,
        era5_to_metar_k=k,
    )
    # Each of the n_metar stations connects to exactly k ERA5 nodes → n_metar * k edges
    assert graph["era5", "to", "metar"].edge_index.shape[1] == n_metar * k


def test_graph_stores_pos_tensors():
    """Every node type must have a .pos tensor of shape (N, 2) with lat/lon."""
    graph = build_heterogeneous_graph(
        era5_nodes=[
            {"lat": 15.0, "lon": 102.0, "feats": [1.0] * 7},
            {"lat": 15.5, "lon": 102.5, "feats": [1.0] * 7},
        ],
        metar_nodes=[{"lat": 15.25, "lon": 102.3, "feats": [0.5] * 5}],
        farm_nodes=[{"lat": 15.1, "lon": 102.2, "farm_id": "f1"}],
        edge_radius_km=200,
    )
    assert hasattr(graph["era5"], "pos"), "era5 nodes missing .pos"
    assert hasattr(graph["metar"], "pos"), "metar nodes missing .pos"
    assert hasattr(graph["farm"], "pos"), "farm nodes missing .pos"

    assert graph["era5"].pos.shape == (2, 2)
    assert graph["metar"].pos.shape == (1, 2)
    assert graph["farm"].pos.shape == (1, 2)

    # First era5 node should store (lat=15.0, lon=102.0)
    assert torch.allclose(graph["era5"].pos[0], torch.tensor([15.0, 102.0]))


def test_radius_edges_exclude_distant_metar_to_farm():
    """metar→farm and era5→farm still use radius — distant pairs get 0 edges."""
    graph = build_heterogeneous_graph(
        era5_nodes=[{"lat": 15.0, "lon": 102.0, "feats": [1.0] * 7}],
        metar_nodes=[{"lat": 1.0, "lon": 104.0, "feats": [0.5] * 5}],
        farm_nodes=[{"lat": 15.1, "lon": 102.3, "farm_id": "f1"}],
        edge_radius_km=100,
    )
    # metar is ~1600 km from the farm — must produce 0 metar→farm edges
    assert graph["metar", "to", "farm"].edge_index.shape[1] == 0
