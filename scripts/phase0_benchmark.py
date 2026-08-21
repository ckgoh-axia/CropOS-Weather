#!/usr/bin/env python3
"""Phase 0 — measure the bar before building anything.

Scores the free public rain forecasts (raw GFS and operational GraphCast-GFS)
at the 16 Thai METAR stations at 24 h and 48 h lead. That score is the bar the
redesigned model must beat, and it gates the grid download.

Every forecast is fetched through select_run(), so the publication-latency
rule in spec 4.3 is enforced for the benchmark exactly as it will be at
inference. A benchmark that cheats would set the bar too high.

Usage:
    PYTHONPATH=. python scripts/phase0_benchmark.py \\
        --labels data/raw/era5_recent.parquet \\
        --metar  data/raw/metar_thai.parquet \\
        --out    data/phase0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.metrics import (
    brier_skill_score,
    brier_skill_score_vs_reference,
)
from src.ingestion.gfs_grib import bucket_start, gfs_key, graphcast_key, select_run
from src.ingestion.grib_fetch import BUCKET_GFS, BUCKET_GRAPHCAST, fetch_point_values
from src.ingestion.metar import STATION_COORDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HORIZONS_H = [24, 48]
THRESHOLD_MM = 1.0
ACCUM_WINDOW_H = 6

# GraphCast-GFS archive begins here; the fine-tune window starts with it.
FIT_START = pd.Timestamp("2024-02-05", tz="UTC")
FIT_END = pd.Timestamp("2025-06-30", tz="UTC")

# Gate: if the best calibrated prior is at or below this at 48 h, there is too
# little skill to correct and the design must be revisited (spec 8).
GATE_MIN_BSS_48H = 0.0


def _apcp_pattern(lead_h: int) -> str:
    """Exact APCP descriptor for the 6-hour bucket ending at lead_h."""
    start = bucket_start(lead_h)
    if lead_h - start != ACCUM_WINDOW_H:
        raise ValueError(
            f"lead {lead_h} is not the end of a {ACCUM_WINDOW_H}h bucket "
            f"(bucket starts at {start})"
        )
    return f"APCP:surface:{start}-{lead_h} hour acc fcst"


def fetch_forecasts(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch GFS and GraphCast-GFS accumulated precip for each day and horizon."""
    rows: list[dict] = []
    for day in pd.date_range(start, end, freq="D", tz="UTC"):
        issuance = day  # 00Z issuance
        for h in HORIZONS_H:
            valid = issuance + pd.Timedelta(hours=h)
            try:
                run, lead = select_run(issuance, valid)
            except ValueError as exc:
                logger.warning(f"skip {issuance} h={h}: {exc}")
                continue
            if lead - bucket_start(lead) != ACCUM_WINDOW_H:
                # select_run may return a lead that is not a 6h bucket end
                # (e.g. 30). Round the issuance back so it is.
                logger.debug(f"skip {issuance} h={h}: lead {lead} not a 6h bucket end")
                continue
            pattern = _apcp_pattern(lead)
            for model, bucket, keyfn in (
                ("gfs", BUCKET_GFS, gfs_key),
                ("graphcast", BUCKET_GRAPHCAST, graphcast_key),
            ):
                try:
                    vals = fetch_point_values(
                        bucket, keyfn(run, lead), pattern, STATION_COORDS
                    )
                except Exception as exc:  # noqa: BLE001 — log and continue
                    logger.warning(f"{model} {run} f{lead:03d}: {exc}")
                    continue
                for station, mm in vals.items():
                    rows.append(
                        {
                            "issuance": issuance,
                            "valid_time": valid,
                            "horizon_h": h,
                            "model": model,
                            "run": run,
                            "lead_h": lead,
                            "station": station,
                            "precip_mm": mm,
                        }
                    )
        logger.info(f"{day.date()}: {len(rows):,} rows so far")
    return pd.DataFrame(rows)


def _read_era5_window(path: Path, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """Read only the ERA5 rows and columns needed for labels.

    era5_recent.parquet is ~60.7M rows; an unfiltered read exhausts RAM on a
    laptop. The filter is strictly narrowing — it cannot change label values.
    """
    return pq.read_table(
        path,
        columns=["timestamp", "lat", "lon", "precipitation"],
        filters=[("timestamp", ">=", lo), ("timestamp", "<=", hi)],
    ).to_pandas()


def build_labels(
    era5_path: Path, metar_path: Path, valid_times: pd.Series
) -> pd.DataFrame:
    """Build 6-hour-accumulated rain labels from ERA5 and METAR.

    Returns one row per (valid_time, station) with columns:
        era5_rain, metar_rain, metar_valid
    """
    from src.features.dataset import _build_era5_label_df

    lo = min(valid_times) - pd.Timedelta(hours=ACCUM_WINDOW_H)
    hi = max(valid_times)
    era5 = _read_era5_window(era5_path, lo, hi)
    era5_lbl = _build_era5_label_df(era5, STATION_COORDS, list(STATION_COORDS))
    era5_lbl = era5_lbl.set_index(["timestamp", "station"]).sort_index()

    metar = pd.read_parquet(metar_path)
    metar["timestamp"] = pd.to_datetime(metar["timestamp"], utc=True)
    metar["precip_mm"] = pd.to_numeric(metar["precip_mm"], errors="coerce")
    metar_lbl = metar.set_index(["timestamp", "station"])["precip_mm"].sort_index()

    rows: list[dict] = []
    offsets = [pd.Timedelta(hours=k) for k in range(ACCUM_WINDOW_H)]
    for vt in sorted(set(valid_times)):
        window = [vt - o for o in offsets]
        for station in STATION_COORDS:
            e_vals = [
                era5_lbl["precip_mm"].get((w, station), np.nan) for w in window
            ]
            m_vals = [metar_lbl.get((w, station), np.nan) for w in window]
            e_sum = np.nansum(e_vals) if not np.all(np.isnan(e_vals)) else np.nan
            m_obs = ~np.all(np.isnan(m_vals))
            m_sum = np.nansum(m_vals) if m_obs else np.nan
            rows.append(
                {
                    "valid_time": vt,
                    "station": station,
                    "era5_rain": float(e_sum >= THRESHOLD_MM)
                    if np.isfinite(e_sum)
                    else np.nan,
                    "metar_rain": float(m_sum >= THRESHOLD_MM)
                    if np.isfinite(m_sum)
                    else np.nan,
                    "metar_valid": bool(m_obs),
                }
            )
    return pd.DataFrame(rows)


def fit_calibration(precip_mm: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit p(rain) = sigmoid(a * log1p(mm) + b). Returns (a, b).

    This is the same functional form the residual head's prior uses (spec
    3.4), so a and b transfer directly as its initialisation.
    """
    from sklearn.linear_model import LogisticRegression

    x = np.log1p(np.clip(precip_mm, 0, None)).reshape(-1, 1)
    clf = LogisticRegression(max_iter=1000).fit(x, labels.astype(int))
    return float(clf.coef_[0][0]), float(clf.intercept_[0])


def apply_calibration(precip_mm: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * np.log1p(np.clip(precip_mm, 0, None)) + b
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="era5_recent.parquet")
    ap.add_argument("--metar", required=True, help="metar_thai.parquet")
    ap.add_argument("--out", default="data/phase0")
    ap.add_argument("--start", default=str(FIT_START.date()))
    ap.add_argument("--end", default=str(FIT_END.date()))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fc_path = out / "forecasts.parquet"
    if fc_path.exists():
        logger.info(f"reusing {fc_path}")
        fc = pd.read_parquet(fc_path)
    else:
        fc = fetch_forecasts(
            pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
        )
        fc.to_parquet(fc_path, index=False)
    logger.info(f"forecast rows: {len(fc):,}")

    lbl = build_labels(Path(args.labels), Path(args.metar), fc["valid_time"])
    df = fc.merge(lbl, on=["valid_time", "station"], how="inner")

    lines = ["# Phase 0 — Forecast Benchmark", ""]
    lines.append(f"Window: {args.start} to {args.end}")
    lines.append(f"Forecast rows: {len(fc):,}  |  matched to labels: {len(df):,}")
    lines.append("")

    results: dict = {}
    for label_col, mask_col in (("era5_rain", None), ("metar_rain", "metar_valid")):
        lines.append(f"## Scored against `{label_col}`")
        lines.append("")
        lines.append("| model | horizon | n | base rate | BSS vs clim |")
        lines.append("|---|---|---|---|---|")
        for model in ("gfs", "graphcast"):
            for h in HORIZONS_H:
                sub = df[(df.model == model) & (df.horizon_h == h)]
                if mask_col:
                    sub = sub[sub[mask_col]]
                sub = sub[np.isfinite(sub[label_col])]
                if len(sub) < 100:
                    lines.append(f"| {model} | {h}h | {len(sub)} | — | insufficient |")
                    continue
                y = sub[label_col].values.astype(np.float32)
                a, b = fit_calibration(sub.precip_mm.values, y)
                p = apply_calibration(sub.precip_mm.values, a, b)
                bss = float(
                    brier_skill_score(p.reshape(-1, 1), y.reshape(-1, 1))[0]
                )
                results[f"{model}_{h}h_{label_col}"] = {
                    "bss_vs_climatology": bss,
                    "calibration_a": a,
                    "calibration_b": b,
                    "n": int(len(sub)),
                    "base_rate": float(y.mean()),
                }
                lines.append(
                    f"| {model} | {h}h | {len(sub):,} | {y.mean():.3f} | {bss:+.4f} |"
                )
        lines.append("")

    # Head-to-head: does GraphCast-GFS beat raw GFS? If so it is the real bar.
    lines.append("## GraphCast-GFS vs raw GFS (the bar)")
    lines.append("")
    lines.append("| horizon | BSS of GraphCast vs GFS |")
    lines.append("|---|---|")
    for h in HORIZONS_H:
        g = df[(df.model == "gfs") & (df.horizon_h == h)]
        c = df[(df.model == "graphcast") & (df.horizon_h == h)]
        m = g.merge(c, on=["valid_time", "station"], suffixes=("_g", "_c"))
        m = m[np.isfinite(m.era5_rain_g)]
        if len(m) < 100:
            lines.append(f"| {h}h | insufficient overlap ({len(m)}) |")
            continue
        y = m.era5_rain_g.values.astype(np.float32).reshape(-1, 1)
        ag, bg = fit_calibration(m.precip_mm_g.values, y.ravel())
        ac, bc = fit_calibration(m.precip_mm_c.values, y.ravel())
        pg = apply_calibration(m.precip_mm_g.values, ag, bg).reshape(-1, 1)
        pc = apply_calibration(m.precip_mm_c.values, ac, bc).reshape(-1, 1)
        rel = float(brier_skill_score_vs_reference(pc, pg, y)[0])
        results[f"graphcast_vs_gfs_{h}h"] = rel
        lines.append(f"| {h}h | {rel:+.4f} |")
    lines.append("")

    # Gate
    best_48 = max(
        results.get("gfs_48h_era5_rain", {}).get("bss_vs_climatology", -9),
        results.get("graphcast_48h_era5_rain", {}).get("bss_vs_climatology", -9),
    )
    passed = best_48 > GATE_MIN_BSS_48H
    lines.append("## Gate")
    lines.append("")
    lines.append(f"- Best calibrated prior BSS at 48 h: **{best_48:+.4f}**")
    lines.append(f"- Threshold: > {GATE_MIN_BSS_48H}")
    lines.append(f"- **{'PASS — proceed' if passed else 'FAIL — revisit design'}**")
    if not passed:
        lines.append("")
        lines.append(
            "The prior has too little skill at 48 h to be worth correcting. "
            "Do NOT commission the grid download. Report and revisit."
        )

    (out / "report.md").write_text("\n".join(lines))
    (out / "gate.json").write_text(
        json.dumps({"passed": passed, "best_bss_48h": best_48, "results": results}, indent=2)
    )
    logger.info(f"wrote {out/'report.md'} and {out/'gate.json'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
