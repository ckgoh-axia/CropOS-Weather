# tests/integration/test_pipeline.py
"""
End-to-end pipeline smoke test.
Runs without real data: uses in-memory stubs to verify all components wire together.
Takes ~10 seconds on CPU, no GPU needed.
"""
from unittest.mock import patch

import numpy as np
import torch

from src.features.graph_builder import build_heterogeneous_graph
from src.ingestion.metar import parse_metar_response
from src.models.gnn import CropOSGNN
from src.preprocessing.qc import flag_metar_outliers
from src.training.evaluate import compute_skill_report
from src.training.loss import BrierCSILoss

SAMPLE_METAR = """station,valid,tmpf,dwpf,relh,drct,sknt,p01i,alti,mslp,vsby,skyc1,wxcodes
VTUU,2023-06-01 00:00,82.4,75.2,80,180,10,0.00,29.85,1010.0,6.00,FEW,
VTUU,2023-06-01 01:00,80.6,74.3,84,190,8,0.12,29.84,1009.8,4.00,OVC,RA
VTUU,2023-06-01 02:00,79.0,73.0,86,185,6,0.08,29.83,1009.5,3.00,OVC,RA
"""


def test_full_pipeline_metar_to_graph_to_gnn():
    # 1. Parse METAR
    metar_df = parse_metar_response(SAMPLE_METAR)
    assert len(metar_df) == 3
    assert "rain_event" in metar_df.columns

    # 2. QC
    qc_df = flag_metar_outliers(metar_df)
    assert "qc_flag" in qc_df.columns
    assert all(qc_df["qc_flag"] == "ok")

    # 3. Build graph
    era5_nodes = [{"lat": 15.0, "lon": 104.0, "feats": [28.0, 26.0, 80.0, 0.0, 8.0, 180.0, 1010.0]}]
    metar_nodes = [{"lat": 15.25, "lon": 104.87, "feats": [3.05, 1.0, 80.6, 74.3, 8.0]}]
    farm_nodes = [{"lat": 15.1, "lon": 104.5, "farm_id": "test_farm"}]
    graph = build_heterogeneous_graph(era5_nodes, metar_nodes, farm_nodes, edge_radius_km=200)
    assert "farm" in graph.node_types

    # 4. GNN forward pass
    model = CropOSGNN(era5_in=7, metar_in=5, hidden=16, n_horizons=4)
    model.eval()
    with torch.no_grad():
        out = model(graph)
    assert out.shape == (1, 4)
    assert (out >= 0).all() and (out <= 1).all()

    # 5. Loss computes without error
    target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    loss_fn = BrierCSILoss()
    loss = loss_fn(out, target)
    assert loss.item() >= 0

    # 6. Skill metrics compute
    report = compute_skill_report(
        model_probs=out.numpy().flatten(),
        nwp_probs=np.array([0.5, 0.5, 0.5, 0.5]),
        observed=target.numpy().flatten(),
    )
    assert "brier_skill_score" in report


def test_api_predict_endpoint_returns_valid_response():
    """API returns well-formed PredictResponse for a Thai farm coordinate."""
    from fastapi.testclient import TestClient

    from src.api.main import app
    client = TestClient(app)
    with patch("src.api.main.run_inference", return_value=torch.tensor([[0.3, 0.7, 0.8, 0.6]])):
        resp = client.post("/predict", json={
            "farm_id": "integration_farm_001",
            "lat": 15.1, "lon": 104.5,
            "timestamp": "2023-06-01T06:00:00Z",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["farm_id"] == "integration_farm_001"
    assert len(data["forecast"]) == 4
    assert data["forecast"][1]["horizon_hours"] == 24
    assert data["forecast"][1]["alert"]  # 0.7 >= 0.5 threshold
