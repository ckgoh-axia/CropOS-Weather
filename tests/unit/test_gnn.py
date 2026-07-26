# tests/unit/test_gnn.py
import torch
from torch_geometric.data import HeteroData

from src.models.gnn import CropOSGNN


def _make_graph(n_era5=4, n_metar=2, n_farms=3):
    data = HeteroData()
    data["era5"].x = torch.randn(n_era5, 7)
    data["metar"].x = torch.randn(n_metar, 5)
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
    out = model(_make_graph())
    assert (out >= 0).all() and (out <= 1).all()

def test_gnn_gradients_flow():
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=32, n_horizons=4)
    out = model(_make_graph())
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
