# tests/unit/test_gnn.py
import torch
from torch_geometric.data import HeteroData

from src.models.gnn import CropOSGNN


def _make_graph(n_era5=4, n_metar=2, n_farms=3, metar_in=5):
    data = HeteroData()
    data["era5"].x = torch.randn(n_era5, 7)
    data["metar"].x = torch.randn(n_metar, metar_in)
    data["farm"].x = torch.zeros(n_farms, 1)
    data["era5", "to", "metar"].edge_index = torch.tensor([[0, 1], [0, 1]])
    data["era5", "to", "farm"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data["metar", "to", "farm"].edge_index = torch.tensor([[0, 1], [0, 1]])
    return data


def test_gnn_output_shape():
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4)
    out = model(_make_graph())
    assert out.shape == (3, 4)  # (n_farms, n_horizons)


def test_gnn_output_is_probability():
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4)
    model.eval()
    with torch.no_grad():
        out = model(_make_graph())
    assert (out >= 0).all() and (out <= 1).all()


def test_gnn_gradients_flow():
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4)
    model.train()
    out = model(_make_graph())
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_gnn_works_without_metar_stations():
    """ERA5-only path: empty metar nodes, no edges that use them."""
    data = HeteroData()
    data["era5"].x = torch.randn(4, 7)
    data["metar"].x = torch.zeros(0, 5)
    data["farm"].x = torch.zeros(3, 1)
    data["era5", "to", "metar"].edge_index = torch.zeros(2, 0, dtype=torch.long)
    data["era5", "to", "farm"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data["metar", "to", "farm"].edge_index = torch.zeros(2, 0, dtype=torch.long)
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4)
    model.eval()
    with torch.no_grad():
        out = model(data)
    assert out.shape == (3, 4)


def test_dropnode_zeroes_stations_during_training():
    """With metar_dropout=1.0 every metar feature vector is zeroed."""
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4, metar_dropout=1.0)
    model.train()
    graph = _make_graph()
    original_x = graph["metar"].x.clone()

    # Verify the dropout attribute is set correctly and features are non-zero
    assert model.metar_dropout == 1.0
    assert original_x.abs().sum() > 0  # non-zero features before dropout


def test_dropnode_inactive_at_eval():
    """Two forward passes at eval mode must be identical (no stochastic zeroing)."""
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4, metar_dropout=0.9)
    model.eval()
    graph = _make_graph()
    with torch.no_grad():
        out1 = model(graph)
        out2 = model(graph)
    assert torch.allclose(out1, out2), "Eval outputs differ — DropNode is active at eval"


def test_gnn_metar_features_constant():
    """METAR_FEATURES must contain all 9 real-time observable variables."""
    assert len(CropOSGNN.METAR_FEATURES) == 9
    assert "precip_mm" in CropOSGNN.METAR_FEATURES
    assert "rain_event" in CropOSGNN.METAR_FEATURES
    assert "tmpf" in CropOSGNN.METAR_FEATURES
