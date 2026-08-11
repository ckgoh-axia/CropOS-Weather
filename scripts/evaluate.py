#!/usr/bin/env python3
"""CropOSGNN full evaluation script.

Usage
-----
    # Evaluate using local parquets (typical post-training use)
    python scripts/evaluate.py --local data/raw

    # Evaluate a specific checkpoint
    python scripts/evaluate.py --local data/raw --checkpoint checkpoints/best_model.pt

    # Evaluate on the held-out test split (2026 H1)
    python scripts/evaluate.py --local data/raw --split test

    # Save detailed CSV reports
    python scripts/evaluate.py --local data/raw --out-dir reports/

Environment
-----------
    HF_TOKEN  — required only when --local is NOT specified (downloads from HuggingFace)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.agri_classifier import AgriClassifier, category_confusion_matrix
from src.evaluation.metrics import (
    brier_skill_score,
    mm_regression_metrics,
    per_horizon_report,
    per_station_report,
)
from src.features.dataset import load_dataset_from_parquets
from src.features.engineer import ERA5_FEATURE_NAMES
from src.ingestion.metar import STATION_COORDS, THAI_METAR_STATIONS
from src.models.gnn import CropOSGNN
from torch_geometric.loader import DataLoader


def _load_config(config_dir: str) -> tuple[dict, dict, dict, dict]:
    with open(f"{config_dir}/model.yaml") as f:
        mcfg = yaml.safe_load(f)
    with open(f"{config_dir}/training.yaml") as f:
        tcfg = yaml.safe_load(f)
    with open(f"{config_dir}/data.yaml") as f:
        dcfg = yaml.safe_load(f)
    eval_path = f"{config_dir}/evaluation.yaml"
    ecfg: dict = {}
    if Path(eval_path).exists():
        with open(eval_path) as f:
            ecfg = yaml.safe_load(f) or {}
    return mcfg, tcfg, dcfg, ecfg


def _run_inference(
    model: CropOSGNN,
    loader: DataLoader,
    device: torch.device,
    dual_head: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run full inference over a DataLoader.

    Returns:
        probs_all:  (N, n_horizons)
        labels_all: (N, n_horizons)
        mm_pred_all:(N, n_horizons)  — zeros if no regression head
        mm_true_all:(N, n_horizons)  — continuous ERA5 mm from dataset
    """
    model.eval()
    all_probs, all_labels, all_mm_pred, all_mm_true = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            if dual_head and isinstance(out, tuple):
                probs, mm_pred = out
            else:
                probs = out
                mm_pred = torch.zeros_like(probs)

            labels = batch["farm"].y   # (n_farms, n_horizons) binary

            # Continuous mm labels (added in Task 2)
            if hasattr(batch["farm"], "precip_mm"):
                mm_true = batch["farm"].precip_mm
            else:
                mm_true = torch.zeros_like(probs)

            all_probs.append(probs.cpu().float().numpy())
            all_labels.append(labels.cpu().float().numpy())
            all_mm_pred.append(mm_pred.cpu().float().numpy())
            all_mm_true.append(mm_true.cpu().float().numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_mm_pred),
        np.concatenate(all_mm_true),
    )


def _reshape_by_station(
    arr: np.ndarray,
    n_stations: int,
    n_horizons: int,
) -> np.ndarray:
    """Reshape (N*n_stations, n_horizons) → (N, n_stations, n_horizons)."""
    n_ts = arr.shape[0] // n_stations
    return arr.reshape(n_ts, n_stations, n_horizons)


def _print_section(title: str) -> None:
    width = 72
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _print_horizon_table(
    report: dict,
    horizons_h: list[int],
    nwp_bss: dict | None,
) -> None:
    header = f"{'Horizon':>8} {'BSS':>7} {'NWP_BSS':>8} {'AUC':>6} {'CSI':>6} {'POD':>6} {'FAR':>6} {'F1':>6} {'RainFrac':>9}"
    print(header)
    print("─" * len(header))
    for h in horizons_h:
        row = report[h]
        nwp = nwp_bss.get(h, nwp_bss.get(str(h), "—")) if nwp_bss else "—"
        nwp_str = f"{float(nwp):7.3f}" if nwp != "—" else f"{'—':>7}"
        bss_indicator = "✓" if row["bss"] > 0 else "✗"
        print(
            f"{h:>6}h   {row['bss']:>+7.3f}{bss_indicator} {nwp_str}"
            f"  {row['auc']:>5.3f}  {row['csi']:>5.3f}"
            f"  {row['pod']:>5.3f}  {row['far']:>5.3f}"
            f"  {row['f1']:>5.3f}  {row['rain_frac']:>8.3f}"
        )


def _print_confusion(report: dict, horizons_h: list[int]) -> None:
    print(f"\n{'':12} {'Predicted NO RAIN':>18}  {'Predicted RAIN':>18}")
    print(f"{'':12} {'─'*18}  {'─'*18}")
    for h in horizons_h:
        row = report[h]
        tn, fp, fn, tp = row["tn"], row["fp"], row["fn"], row["tp"]
        print(f"  t+{h:02d}h")
        print(f"  Actual NO  │   TN={tn:>6.0f}            FP={fp:>6.0f}  (FAR={row['far']:.2f})")
        print(f"  Actual YES │   FN={fn:>6.0f} MISS       TP={tp:>6.0f}  (POD={row['pod']:.2f})")
        print(f"  {'─'*60}")


def evaluate(
    config_dir: str = "configs",
    checkpoint_path: str = "checkpoints/best_model.pt",
    local_data_dir: str | None = None,
    split: str = "val",
    out_dir: str | None = None,
    decision_threshold: float = 0.5,
) -> None:
    mcfg, tcfg, dcfg, ecfg = _load_config(config_dir)
    gnn_cfg = mcfg["gnn"]
    horizons_h: list[int] = dcfg["forecast_horizons"]
    threshold_mm: float = dcfg["precipitation_label_threshold_mm"]
    era5_radius_km: float = gnn_cfg.get("edge_radius_km", 100.0)

    # Date range for split
    if split == "val":
        start_date = dcfg["validation_start"]
        end_date = dcfg["validation_end"]
    elif split == "test":
        start_date = dcfg["test_start"]
        end_date = dcfg["test_end"]
    else:
        raise ValueError(f"Unknown split: {split!r}. Use 'val' or 'test'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Split: {split}  ({start_date} → {end_date})")
    print(f"Checkpoint: {checkpoint_path}")

    # ── load dataset ─────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    if local_data_dir:
        data_dir = Path(local_data_dir)
        ds = load_dataset_from_parquets(
            era5_path=data_dir / "era5_thailand.parquet",
            metar_path=data_dir / "metar_thai.parquet",
            station_order=THAI_METAR_STATIONS,
            station_coords=STATION_COORDS,
            start_date=start_date,
            end_date=end_date,
            era5_node_radius_km=era5_radius_km,
            horizons_h=horizons_h,
            threshold_mm=threshold_mm,
            era5_north_path=data_dir / "era5_north.parquet" if (data_dir / "era5_north.parquet").exists() else None,
            era5_recent_path=data_dir / "era5_recent.parquet" if (data_dir / "era5_recent.parquet").exists() else None,
        )
    else:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("Set HF_TOKEN or use --local <dir>")
        from huggingface_hub import HfApi
        username = HfApi().whoami(token=hf_token)["name"]
        repo_id = f"{username}/cropos-data"
        from src.features.dataset import load_dataset_from_hf
        ds = load_dataset_from_hf(
            repo_id=repo_id,
            hf_token=hf_token,
            station_order=THAI_METAR_STATIONS,
            station_coords=STATION_COORDS,
            start_date=start_date,
            end_date=end_date,
            era5_node_radius_km=era5_radius_km,
            horizons_h=horizons_h,
            threshold_mm=threshold_mm,
        )

    if len(ds) == 0:
        print("ERROR: evaluation dataset is empty. Check date range and data files.")
        sys.exit(1)

    print(f"Loaded {len(ds):,} evaluation timestamps")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    # ── load model ────────────────────────────────────────────────────────────
    era5_in = gnn_cfg.get("era5_in", len(ERA5_FEATURE_NAMES))
    metar_in = gnn_cfg.get("metar_in", 9)
    dual_head = gnn_cfg.get("dual_head", False)

    model = CropOSGNN(
        era5_in=era5_in,
        hidden=gnn_cfg["hidden_channels"],
        n_horizons=len(horizons_h),
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
        metar_dropout=0.0,   # no dropout at eval
        metar_in=metar_in,
        dual_head=dual_head,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handle both raw state_dict and wrapped checkpoint dicts
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=False)
    model.eval()

    # ── run inference ─────────────────────────────────────────────────────────
    print("Running inference...")
    probs_all, labels_all, mm_pred_all, mm_true_all = _run_inference(model, loader, device, dual_head)

    n_samples = probs_all.shape[0]
    n_stations = len(THAI_METAR_STATIONS)
    n_horizons = len(horizons_h)

    # Reshape flat (N*S, H) → (N, S, H) for per-station analysis
    probs_3d = _reshape_by_station(probs_all, n_stations, n_horizons)
    labels_3d = _reshape_by_station(labels_all, n_stations, n_horizons)

    # ── metrics ───────────────────────────────────────────────────────────────
    nwp_benchmark = ecfg.get("nwp_bss_benchmark", {})
    prob_threshold = ecfg.get("thresholds", {}).get("rain_probability", decision_threshold)

    horizon_rpt = per_horizon_report(probs_all, labels_all, horizons_h, threshold=prob_threshold)
    station_rpt = per_station_report(probs_3d, labels_3d, THAI_METAR_STATIONS, horizons_h, threshold=prob_threshold)

    # ── print report ──────────────────────────────────────────────────────────
    _print_section("CROPOS GNN — EVALUATION REPORT")
    print(f"  Split:      {split}  ({start_date} → {end_date})")
    print(f"  Samples:    {n_samples:,} (farm-timestamp pairs)")
    print(f"  Horizons:   {horizons_h} hours")
    print(f"  Threshold:  P(rain) ≥ {prob_threshold:.2f}")

    # Overall rain fraction
    overall_rain = float(labels_all.mean())
    print(f"  Rain freq:  {overall_rain:.3f} ({overall_rain*100:.1f}% of labels are rain events)")

    _print_section("BRIER SKILL SCORE  (positive = better than climatology)")
    _print_horizon_table(horizon_rpt, horizons_h, nwp_benchmark)

    _print_section("CONFUSION MATRIX  per horizon")
    _print_confusion(horizon_rpt, horizons_h)

    # Critical agricultural metrics: false negatives = silent misses (dangerous for farming)
    _print_section("AGRICULTURAL RISK SUMMARY")
    for h in horizons_h:
        row = horizon_rpt[h]
        fn_pct = row["miss_rate"] * 100
        fp_pct = row["far"] * 100
        n_fn = int(row["fn"])
        n_fp = int(row["fp"])
        print(
            f"  t+{h:02d}h │ Silent misses (FN): {fn_pct:5.1f}%  ({n_fn:,} rain events predicted as dry)"
            f"  │ False alarms (FP): {fp_pct:5.1f}% ({n_fp:,})"
        )

    _print_section("MM REGRESSION METRICS (only meaningful with dual_head=True)")
    if dual_head and mm_pred_all.max() > 0:
        mm_rpt = mm_regression_metrics(mm_pred_all, mm_true_all, horizons_h)
        print(f"  {'Horizon':>8}  {'RMSE':>7}  {'MAE':>7}  {'Bias':>8}")
        for h in horizons_h:
            r = mm_rpt[h]
            print(f"  {h:>6}h    {r['rmse']:>6.2f}   {r['mae']:>6.2f}  {r['bias']:>+8.2f}")
    else:
        print("  Skipped — model has no regression head.")
        print("  Add 'dual_head: true' to configs/model.yaml and retrain.")

    _print_section("AGRICULTURAL CATEGORY CONFUSION  (pred vs actual, 24h horizon)")
    h24_idx = horizons_h.index(24) if 24 in horizons_h else 0
    if dual_head and mm_pred_all.max() > 0:
        agri_cm = category_confusion_matrix(mm_pred_all[:, h24_idx], mm_true_all[:, h24_idx])
    else:
        # No regression head — estimate mm from probability using scale
        clf = AgriClassifier()
        est_mm = probs_all[:, h24_idx] * clf.prob_to_mm_scale
        agri_cm = category_confusion_matrix(est_mm, mm_true_all[:, h24_idx])
        print("  (mm estimated from probability × 15.0 — enable dual_head for true mm)")
    # Print non-zero cells
    for (pred_cat, true_cat), count in sorted(agri_cm.items(), key=lambda x: -x[1]):
        match = "✓" if pred_cat == true_cat else "✗"
        if count > 0:
            print(f"  {match} Predicted {pred_cat:<16} | Actual {true_cat:<16}: {count:,}")

    _print_section("PER-STATION BSS  (24h horizon)")
    h24_val = horizons_h[h24_idx]
    print(f"  {'Station':>8}  {'BSS':>7}  {'POD':>6}  {'FAR':>6}  {'Rain%':>6}")
    for station in THAI_METAR_STATIONS:
        row = station_rpt[station][h24_val]
        bss_flag = "✓" if row["bss"] > 0 else "✗"
        print(
            f"  {station:>8}  {row['bss']:>+6.3f}{bss_flag}"
            f"  {row['pod']:>5.3f}  {row['far']:>5.3f}  {row['rain_frac']:>5.1%}"
        )

    # ── optional CSV export ───────────────────────────────────────────────────
    if out_dir:
        import pandas as pd
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        rows = []
        for h, rpt in horizon_rpt.items():
            rows.append({"horizon_h": h, **rpt})
        pd.DataFrame(rows).to_csv(out / f"horizon_metrics_{split}.csv", index=False)

        station_rows = []
        for station, h_rpt in station_rpt.items():
            for h, rpt in h_rpt.items():
                station_rows.append({"station": station, "horizon_h": h, **rpt})
        pd.DataFrame(station_rows).to_csv(out / f"station_metrics_{split}.csv", index=False)

        print(f"\nCSV reports saved to {out_dir}/")

    print(f"\n{'═' * 72}")
    print("  Evaluation complete.")
    print(f"{'═' * 72}\n")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CropOSGNN")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--local", metavar="DIR", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out-dir", default=None, help="Save CSV reports here")
    parser.add_argument("--threshold", type=float, default=0.5, help="Rain probability threshold")
    args = parser.parse_args()
    evaluate(
        config_dir=args.config_dir,
        checkpoint_path=args.checkpoint,
        local_data_dir=args.local,
        split=args.split,
        out_dir=args.out_dir,
        decision_threshold=args.threshold,
    )


if __name__ == "__main__":
    _cli()
