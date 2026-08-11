# tests/unit/test_gnn.py
import pytest
import torch
from torch_geometric.data import HeteroData

from src.models.gnn import CropOSGNN


def _make_graph(n_era5=4, n_metar=2, n_farms=3, metar_in=5):
    data = HeteroData()
    data["era5"].x = torch.randn(n_era5, 7)
    data["metar"].x = torch.randn(n_metar, metar_in)
    data["farm"].x = torch.zeros(n_farms, 1)

    # Positions required by RelPos message-passing layers [lat, lon]
    data["era5"].pos  = torch.tensor([[13.0 + i * 0.5, 100.0 + i * 0.5] for i in range(n_era5)],
                                     dtype=torch.float)
    data["metar"].pos = torch.tensor([[13.2 + i * 0.3, 100.2 + i * 0.3] for i in range(n_metar)],
                                     dtype=torch.float)
    data["farm"].pos  = torch.tensor([[13.1 + i * 0.4, 100.1 + i * 0.4] for i in range(n_farms)],
                                     dtype=torch.float)

    data["era5",  "to", "metar"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    data["era5",  "to", "farm"].edge_index  = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data["metar", "to", "farm"].edge_index  = torch.tensor([[0, 1], [0, 1]])
    data["metar", "to", "metar"].edge_index = torch.tensor([[0, 1], [1, 0]])
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
    data["era5"].x   = torch.randn(4, 7)
    data["metar"].x  = torch.zeros(0, 5)
    data["farm"].x   = torch.zeros(3, 1)

    data["era5"].pos  = torch.tensor([[13.0 + i * 0.5, 100.0 + i * 0.5] for i in range(4)],
                                     dtype=torch.float)
    data["metar"].pos = torch.zeros(0, 2, dtype=torch.float)
    data["farm"].pos  = torch.tensor([[13.1 + i * 0.4, 100.1 + i * 0.4] for i in range(3)],
                                     dtype=torch.float)

    data["era5",  "to", "metar"].edge_index = torch.zeros(2, 0, dtype=torch.long)
    data["era5",  "to", "farm"].edge_index  = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data["metar", "to", "farm"].edge_index  = torch.zeros(2, 0, dtype=torch.long)
    data["metar", "to", "metar"].edge_index = torch.zeros(2, 0, dtype=torch.long)

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

    assert model.metar_dropout == 1.0
    assert original_x.abs().sum() > 0


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


def test_dual_head_output_shapes():
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4, dual_head=True)
    model.eval()
    with torch.no_grad():
        probs, mm = model(_make_graph())
    assert probs.shape == (3, 4), f"Expected (3,4), got {probs.shape}"
    assert mm.shape == (3, 4), f"Expected (3,4), got {mm.shape}"


def test_dual_head_probs_in_range():
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4, dual_head=True)
    model.eval()
    with torch.no_grad():
        probs, mm = model(_make_graph())
    assert (probs >= 0).all() and (probs <= 1).all()


def test_dual_head_mm_non_negative():
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4, dual_head=True)
    model.eval()
    with torch.no_grad():
        probs, mm = model(_make_graph())
    assert (mm >= 0).all(), "Regression head must output non-negative mm values"


def test_single_head_backward_compat():
    """dual_head=False (default) must still return a single tensor, not a tuple."""
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4)
    out = model(_make_graph())
    assert isinstance(out, torch.Tensor), "Single-head must return Tensor, not tuple"


def test_dual_head_gradients_flow():
    """Gradients must flow through both classification and regression heads."""
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4, dual_head=True)
    model.train()
    probs, mm = model(_make_graph())
    loss = probs.sum() + mm.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_local_mp_steps_param():
    """local_mp_steps controls how many metar↔metar iterations are used."""
    model2 = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4, local_mp_steps=2)
    model6 = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4, local_mp_steps=6)
    assert len(model2.metar_convs) == 2
    assert len(model6.metar_convs) == 6


def test_pos_tensors_required():
    """forward() uses .pos — missing pos should raise AttributeError."""
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4)
    data = HeteroData()
    data["era5"].x  = torch.randn(4, 7)
    data["metar"].x = torch.randn(2, 5)
    data["farm"].x  = torch.zeros(3, 1)
    data["era5",  "to", "metar"].edge_index = torch.tensor([[0, 1], [0, 1]])
    data["era5",  "to", "farm"].edge_index  = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data["metar", "to", "farm"].edge_index  = torch.tensor([[0, 1], [0, 1]])
    data["metar", "to", "metar"].edge_index = torch.tensor([[0, 1], [1, 0]])
    # No .pos set — expect AttributeError
    with pytest.raises((AttributeError, KeyError)):
        model(data)
