# Phase 0 — Forecast Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the Brier Skill Score of the best free public rain forecast (raw GFS and operational GraphCast-GFS) at the 16 Thai METAR stations at 24 h and 48 h lead, to establish the bar the redesigned model must beat — and to gate the expensive grid download.

**Architecture:** A leakage-safe GRIB point-extraction layer over NOAA's public S3 buckets, addressed by explicit model run and lead hour. Run selection enforces publication latency as a pure, unit-tested function. A scoring script aligns forecasts to ERA5 and METAR labels, fits a 1-D calibration on the fine-tune window only, and emits a report plus a machine-readable gate verdict.

**Tech Stack:** Python 3.11, pandas, numpy, xarray + cfgrib (eccodes), requests (HTTP byte-range), pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-cropos-gfs-hybrid-design.md` (revision 2)

## Global Constraints

- **Leakage rule (spec §4.3):** every forecast field must come from a model run whose *publication* time is ≤ issuance time. GFS publishes ~4 h after initialisation. `PUBLICATION_LAG_H = 4`.
- **Horizons:** 24 h and 48 h only. No 12 h or 36 h.
- **Accumulation:** GFS `APCP` accumulates from the last multiple of 6. `bucket_start(h) = 6 * floor((h - 1) / 6)`. At f024 the bucket is 18–24 h; at f025 it is 24–25 h. Never mix buckets of different widths.
- **Phase 0 target window:** rain ≥ 1.0 mm accumulated in the **6-hour window ending at the horizon**. This is the finest resolution both GFS and GraphCast-GFS support natively (GraphCast-GFS emits only f006/f012/f018/f024/...).
- **Calibration is fitted on the fine-tune window (2024-02-05 → 2025-06-30) only.** Never on validation or test.
- **Test window (2026-01-01 → 2026-06-30) is not touched by this plan.**
- **Stations:** the 16 in `src.ingestion.metar.STATION_COORDS`. Do not redefine them.
- **Line length 100** (`ruff`, per `pyproject.toml`). Lint must pass.
- Run tests with `PYTHONPATH=. pytest tests/unit/ -v`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ingestion/gfs_grib.py` | **New.** Run selection (leakage guard), S3 key construction, APCP bucket arithmetic. Pure logic, no network. |
| `src/ingestion/grib_fetch.py` | **New.** HTTP `.idx` parsing, byte-range GRIB retrieval, decode to point values. All network I/O lives here. |
| `scripts/phase0_benchmark.py` | **New.** Orchestration: fetch → align → calibrate → score → report → gate verdict. |
| `src/evaluation/metrics.py` | **Modify.** Add `brier_skill_score_vs_reference`. |
| `tests/unit/test_gfs_grib.py` | **New.** Leakage guard and bucket arithmetic. Pure, no network. |
| `tests/unit/test_grib_fetch.py` | **New.** `.idx` parsing and byte-range construction, mocked HTTP. |
| `tests/unit/test_metrics.py` | **Modify.** Cover the new reference-BSS function. |

Splitting pure logic (`gfs_grib.py`) from network I/O (`grib_fetch.py`) is deliberate: the leakage guard is the correctness-critical part and must be testable without a network.

---

## Task 1: Leakage guard and GRIB addressing

**Files:**
- Create: `src/ingestion/gfs_grib.py`
- Test: `tests/unit/test_gfs_grib.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PUBLICATION_LAG_H: int = 4`
  - `select_run(issuance: pd.Timestamp, valid_time: pd.Timestamp, publication_lag_h: int = 4) -> tuple[pd.Timestamp, int]` — returns `(run_init_time, lead_hours)`. Raises `ValueError` if no run qualifies.
  - `bucket_start(lead_h: int) -> int`
  - `gfs_key(run: pd.Timestamp, lead_h: int) -> str`
  - `graphcast_key(run: pd.Timestamp, lead_h: int) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_gfs_grib.py`:

```python
# tests/unit/test_gfs_grib.py
import pandas as pd
import pytest

from src.ingestion.gfs_grib import (
    PUBLICATION_LAG_H,
    bucket_start,
    gfs_key,
    graphcast_key,
    select_run,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_publication_lag_is_four_hours():
    assert PUBLICATION_LAG_H == 4


def test_select_run_picks_latest_published_run():
    # Issuance 12:00Z. The 06Z run published at 10:00Z -> usable.
    # The 12Z run publishes at 16:00Z -> NOT usable.
    run, lead = select_run(_ts("2025-03-10 12:00"), _ts("2025-03-11 12:00"))
    assert run == _ts("2025-03-10 06:00")
    assert lead == 30


def test_select_run_excludes_unpublished_run():
    # Issuance 09:00Z: the 06Z run publishes at 10:00Z, so it is NOT yet
    # available. Must fall back to the 00Z run (published 04:00Z).
    run, lead = select_run(_ts("2025-03-10 09:00"), _ts("2025-03-11 09:00"))
    assert run == _ts("2025-03-10 00:00")
    assert lead == 33


def test_select_run_boundary_exactly_at_publication():
    # Issuance exactly 10:00Z == publication time of the 06Z run. Inclusive.
    run, _ = select_run(_ts("2025-03-10 10:00"), _ts("2025-03-11 10:00"))
    assert run == _ts("2025-03-10 06:00")


def test_select_run_rejects_valid_time_before_run():
    with pytest.raises(ValueError):
        select_run(_ts("2025-03-10 12:00"), _ts("2025-03-10 01:00"))


def test_select_run_lead_never_negative_and_run_always_published():
    for hour in range(24):
        issuance = _ts(f"2025-03-10 {hour:02d}:00")
        run, lead = select_run(issuance, issuance + pd.Timedelta(hours=24))
        assert lead > 0
        assert run + pd.Timedelta(hours=PUBLICATION_LAG_H) <= issuance
        assert run.hour % 6 == 0


def test_bucket_start_resets_every_six_hours():
    # APCP accumulates from the last multiple of 6.
    # Verified against real .idx files on noaa-gfs-bdp-pds.
    assert bucket_start(21) == 18   # "18-21 hour acc fcst"
    assert bucket_start(23) == 18   # "18-23 hour acc fcst"
    assert bucket_start(24) == 18   # "18-24 hour acc fcst"
    assert bucket_start(25) == 24   # "24-25 hour acc fcst"
    assert bucket_start(26) == 24   # "24-26 hour acc fcst"
    assert bucket_start(48) == 42   # "42-48 hour acc fcst"


def test_bucket_width_at_horizons_is_six_hours():
    for h in (24, 48):
        assert h - bucket_start(h) == 6


def test_gfs_key_modern_layout():
    key = gfs_key(_ts("2026-08-01 00:00"), 24)
    assert key == "gfs.20260801/00/atmos/gfs.t00z.pgrb2.0p25.f024"


def test_gfs_key_legacy_layout_has_no_atmos_dir():
    # Before 2021-03-23 the bucket has no atmos/ subdirectory.
    key = gfs_key(_ts("2021-03-01 06:00"), 48)
    assert key == "gfs.20210301/06/gfs.t06z.pgrb2.0p25.f048"


def test_graphcast_key_layout():
    key = graphcast_key(_ts("2024-06-01 00:00"), 24)
    assert key == (
        "graphcastgfs.20240601/00/forecasts_13_levels/"
        "graphcastgfs.t00z.pgrb2.0p25.f024"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_gfs_grib.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.gfs_grib'`

- [ ] **Step 3: Write the implementation**

Create `src/ingestion/gfs_grib.py`:

```python
"""GRIB addressing and leakage-safe run selection for NOAA GFS on AWS.

This module is pure logic — no network I/O — because the run-selection rule is
the correctness-critical part of the whole pipeline and must be testable
without a network.

Leakage rule (spec §4.3)
------------------------
Every forecast field must come from a model run whose PUBLICATION time is at
or before the issuance time. Initialisation time is not sufficient: GFS
publishes roughly 4 hours after initialisation, so a run initialised at 06Z is
not available to a forecast issued at 09Z.

Getting this wrong is silent. Validation scores rise and production fails,
because the run the model was trained to expect does not exist yet at
inference time.

Accumulation convention
-----------------------
GFS ``APCP`` accumulates from the last multiple of 6 hours, NOT from the run
start and NOT over a fixed window. Verified against real .idx files:

    f021 -> "18-21 hour acc fcst"   (3 h)
    f024 -> "18-24 hour acc fcst"   (6 h)
    f025 -> "24-25 hour acc fcst"   (1 h)

Mixing these silently compares a 6-hour total against a 1-hour total.
"""
from __future__ import annotations

import pandas as pd

# GFS runs 4x daily and publishes ~4 h after initialisation.
RUN_INTERVAL_H: int = 6
PUBLICATION_LAG_H: int = 4

# The bucket switched to a gfs.YYYYMMDD/HH/atmos/ layout on this date.
_ATMOS_LAYOUT_FROM = pd.Timestamp("2021-03-23", tz="UTC")


def select_run(
    issuance: pd.Timestamp,
    valid_time: pd.Timestamp,
    publication_lag_h: int = PUBLICATION_LAG_H,
) -> tuple[pd.Timestamp, int]:
    """Return the freshest run usable at ``issuance``, and its lead hours.

    Args:
        issuance:   The time the forecast is issued. Only runs published at or
                    before this instant may be used.
        valid_time: The time the forecast is valid for.
        publication_lag_h: Hours between run initialisation and availability.

    Returns:
        (run_init_time, lead_hours) where lead_hours = valid_time - run.

    Raises:
        ValueError: if valid_time is not after the selected run.
    """
    # Latest 6-hourly run whose publication time is <= issuance.
    latest_publishable = issuance - pd.Timedelta(hours=publication_lag_h)
    run = latest_publishable.floor(f"{RUN_INTERVAL_H}h")

    lead = int((valid_time - run).total_seconds() // 3600)
    if lead <= 0:
        raise ValueError(
            f"valid_time {valid_time} is not after selected run {run} "
            f"(issuance {issuance}, lag {publication_lag_h} h)"
        )
    return run, lead


def bucket_start(lead_h: int) -> int:
    """Return the lead hour at which this APCP accumulation bucket started."""
    if lead_h <= 0:
        raise ValueError(f"lead_h must be positive, got {lead_h}")
    return RUN_INTERVAL_H * ((lead_h - 1) // RUN_INTERVAL_H)


def gfs_key(run: pd.Timestamp, lead_h: int) -> str:
    """Return the S3 key for a GFS pgrb2 0.25-degree file."""
    day = run.strftime("%Y%m%d")
    hh = run.strftime("%H")
    name = f"gfs.t{hh}z.pgrb2.0p25.f{lead_h:03d}"
    if run >= _ATMOS_LAYOUT_FROM:
        return f"gfs.{day}/{hh}/atmos/{name}"
    return f"gfs.{day}/{hh}/{name}"


def graphcast_key(run: pd.Timestamp, lead_h: int) -> str:
    """Return the S3 key for an operational GraphCast-GFS pgrb2 file."""
    day = run.strftime("%Y%m%d")
    hh = run.strftime("%H")
    return (
        f"graphcastgfs.{day}/{hh}/forecasts_13_levels/"
        f"graphcastgfs.t{hh}z.pgrb2.0p25.f{lead_h:03d}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_gfs_grib.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Run lint and the full unit suite**

Run: `ruff check src/ingestion/gfs_grib.py tests/unit/test_gfs_grib.py && PYTHONPATH=. pytest tests/unit/ -v`
Expected: no lint errors; all pre-existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/gfs_grib.py tests/unit/test_gfs_grib.py
git commit -m "feat(ingest): leakage-safe GFS run selection and GRIB addressing

select_run() enforces the publication-latency rule from spec 4.3: a run is
usable only once init + 4h <= issuance. bucket_start() encodes the GFS APCP
convention (accumulates from the last multiple of 6), verified against real
.idx files. Pure logic, no network, so the leakage guard is unit-testable."
```

---

## Task 2: GRIB byte-range point extraction

**Files:**
- Create: `src/ingestion/grib_fetch.py`
- Test: `tests/unit/test_grib_fetch.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `gfs_key`, `graphcast_key` from Task 1.
- Produces:
  - `BUCKET_GFS: str`, `BUCKET_GRAPHCAST: str`
  - `parse_idx(idx_text: str) -> list[dict]` — each dict has `msg`, `start`, `stop` (`stop` is `None` for the final message), `descriptor`.
  - `find_message(entries: list[dict], pattern: str) -> dict` — raises `KeyError` if not found or ambiguous.
  - `fetch_point_values(bucket: str, key: str, pattern: str, coords: dict[str, tuple[float, float]]) -> dict[str, float]`

- [ ] **Step 1: Add the GRIB dependency**

Edit `pyproject.toml`, in `[tool.poetry.dependencies]` after the `xarray` line:

```toml
cfgrib = "^0.9.10"
eccodes = "^1.7.0"
```

Install the eccodes binary (cfgrib is only a wrapper):

```bash
# macOS
brew install eccodes
# Debian/Ubuntu, incl. the RunPod image
apt-get update && apt-get install -y libeccodes-dev
```

Then: `poetry lock && poetry install`

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_grib_fetch.py`:

```python
# tests/unit/test_grib_fetch.py
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.grib_fetch import (
    BUCKET_GFS,
    find_message,
    parse_idx,
)

# Verbatim excerpt from
# noaa-gfs-bdp-pds/gfs.20260801/00/atmos/gfs.t00z.pgrb2.0p25.f024.idx
_IDX = """593:419694393:d=2026080100:PRATE:surface:24 hour fcst:
595:421031721:d=2026080100:PRATE:surface:18-24 hour ave fcst:
596:421726683:d=2026080100:APCP:surface:18-24 hour acc fcst:
597:422103206:d=2026080100:APCP:surface:0-1 day acc fcst:
624:431802062:d=2026080100:CAPE:surface:24 hour fcst:
"""


def test_bucket_is_public_noaa_gfs():
    assert BUCKET_GFS == "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


def test_parse_idx_extracts_byte_ranges():
    entries = parse_idx(_IDX)
    assert len(entries) == 5
    assert entries[0]["start"] == 419694393
    # stop is the NEXT message's start offset
    assert entries[0]["stop"] == 421031721


def test_parse_idx_last_message_has_open_ended_range():
    entries = parse_idx(_IDX)
    assert entries[-1]["stop"] is None


def test_find_message_matches_exact_accumulation_window():
    entries = parse_idx(_IDX)
    msg = find_message(entries, "APCP:surface:18-24 hour acc fcst")
    assert msg["start"] == 421726683
    assert msg["stop"] == 422103206


def test_find_message_rejects_ambiguous_pattern():
    # "APCP:surface" alone matches BOTH the 18-24h bucket and the 0-1day
    # total. Silently picking one would mix accumulation windows.
    entries = parse_idx(_IDX)
    with pytest.raises(KeyError, match="ambiguous"):
        find_message(entries, "APCP:surface")


def test_find_message_raises_when_absent():
    entries = parse_idx(_IDX)
    with pytest.raises(KeyError):
        find_message(entries, "NOSUCHVAR:surface:24 hour fcst")


@patch("src.ingestion.grib_fetch.requests.get")
def test_fetch_point_values_requests_only_the_needed_bytes(mock_get):
    from src.ingestion.grib_fetch import fetch_point_values

    idx_response = MagicMock(status_code=200, text=_IDX)
    grib_response = MagicMock(status_code=206, content=b"GRIB-BYTES")
    mock_get.side_effect = [idx_response, grib_response]

    with patch("src.ingestion.grib_fetch._decode_points", return_value={"VTUU": 1.5}):
        out = fetch_point_values(
            BUCKET_GFS,
            "gfs.20260801/00/atmos/gfs.t00z.pgrb2.0p25.f024",
            "APCP:surface:18-24 hour acc fcst",
            {"VTUU": (15.25, 104.87)},
        )

    assert out == {"VTUU": 1.5}
    # Second call must be a bounded Range request, not a full download.
    _, kwargs = mock_get.call_args_list[1]
    assert kwargs["headers"]["Range"] == "bytes=421726683-422103205"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_grib_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.grib_fetch'`

- [ ] **Step 4: Write the implementation**

Create `src/ingestion/grib_fetch.py`:

```python
"""Byte-range GRIB2 retrieval from NOAA's public S3 buckets.

A single GFS pgrb2 file holds ~743 GRIB messages and is several hundred MB.
Every file is accompanied by a .idx sidecar listing each message's byte
offset, so we fetch the index, locate the one message we want, and issue an
HTTP Range request for just those bytes. Downloading whole files instead
would make the backfill impractical.

All network I/O for GRIB lives here; the addressing and leakage rules live in
gfs_grib.py and stay unit-testable without a network.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BUCKET_GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
BUCKET_GRAPHCAST = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"

REQUEST_TIMEOUT_S = 120


def parse_idx(idx_text: str) -> list[dict]:
    """Parse a GRIB .idx sidecar into messages with explicit byte ranges.

    Each .idx line is ``msg:offset:date:VAR:level:fcst-descriptor:``. The end
    of a message is the start of the next one; the final message runs to the
    end of the file, represented as ``stop=None``.
    """
    entries: list[dict] = []
    for line in idx_text.strip().splitlines():
        parts = line.split(":")
        if len(parts) < 6:
            continue
        entries.append(
            {
                "msg": int(parts[0]),
                "start": int(parts[1]),
                "stop": None,
                "descriptor": ":".join(parts[3:6]),
            }
        )
    for i in range(len(entries) - 1):
        entries[i]["stop"] = entries[i + 1]["start"]
    return entries


def find_message(entries: list[dict], pattern: str) -> dict:
    """Return the single message whose descriptor contains ``pattern``.

    Raises KeyError if there is no match, or more than one. Ambiguity is an
    error rather than a first-match, because "APCP:surface" matches both the
    6-hour bucket and the run-total accumulation — silently choosing one
    would mix accumulation windows across the dataset.
    """
    matches = [e for e in entries if pattern in e["descriptor"]]
    if not matches:
        raise KeyError(f"no GRIB message matching {pattern!r}")
    if len(matches) > 1:
        found = [m["descriptor"] for m in matches]
        raise KeyError(f"pattern {pattern!r} is ambiguous, matched: {found}")
    return matches[0]


def _decode_points(
    grib_bytes: bytes, coords: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """Decode a single GRIB message and sample it at the given coordinates."""
    import xarray as xr

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as fh:
        fh.write(grib_bytes)
        tmp = Path(fh.name)
    try:
        ds = xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""})
        var = list(ds.data_vars)[0]
        out: dict[str, float] = {}
        for name, (lat, lon) in coords.items():
            # GFS longitudes run 0-360; Thai stations are all positive east.
            sel = ds[var].sel(
                latitude=lat, longitude=lon % 360, method="nearest"
            )
            out[name] = float(sel.values)
        ds.close()
        return out
    finally:
        tmp.unlink(missing_ok=True)


def fetch_point_values(
    bucket: str,
    key: str,
    pattern: str,
    coords: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Fetch one GRIB message by byte range and sample it at station points.

    Returns {station_id: value}. Raises on HTTP or decode failure — callers
    must not silently substitute zeros for missing data (spec §4.3).
    """
    idx = requests.get(f"{bucket}/{key}.idx", timeout=REQUEST_TIMEOUT_S)
    idx.raise_for_status()
    entries = parse_idx(idx.text)
    msg = find_message(entries, pattern)

    end = "" if msg["stop"] is None else str(msg["stop"] - 1)
    headers = {"Range": f"bytes={msg['start']}-{end}"}
    resp = requests.get(f"{bucket}/{key}", headers=headers, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()

    return _decode_points(resp.content, coords)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_grib_fetch.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 6: Verify against the real bucket once, by hand**

This is a smoke check that the byte ranges and decode actually work end to end.

```bash
PYTHONPATH=. python -c "
from src.ingestion.grib_fetch import BUCKET_GFS, fetch_point_values
from src.ingestion.metar import STATION_COORDS
v = fetch_point_values(
    BUCKET_GFS,
    'gfs.20260801/00/atmos/gfs.t00z.pgrb2.0p25.f024',
    'APCP:surface:18-24 hour acc fcst',
    STATION_COORDS,
)
print(len(v), 'stations')
print({k: round(x, 2) for k, x in list(v.items())[:5]})
assert len(v) == 16
assert all(x >= 0 for x in v.values()), 'precipitation must be non-negative'
"
```
Expected: 16 stations, non-negative mm values.

- [ ] **Step 7: Run lint and the full suite, then commit**

```bash
ruff check src/ingestion/grib_fetch.py tests/unit/test_grib_fetch.py
PYTHONPATH=. pytest tests/unit/ -v
git add pyproject.toml src/ingestion/grib_fetch.py tests/unit/test_grib_fetch.py
git commit -m "feat(ingest): byte-range GRIB point extraction from NOAA S3

Fetches the .idx sidecar, locates one message, and Range-requests only those
bytes rather than the ~500MB file. find_message() treats an ambiguous pattern
as an error: 'APCP:surface' matches both the 6h bucket and the run total, and
silently picking one would mix accumulation windows."
```

---

## Task 3: Brier Skill Score against a reference forecast

**Files:**
- Modify: `src/evaluation/metrics.py`
- Modify: `tests/unit/test_metrics.py`

**Interfaces:**
- Consumes: existing `brier_score` in `src/evaluation/metrics.py`.
- Produces: `brier_skill_score_vs_reference(probs: np.ndarray, reference: np.ndarray, labels: np.ndarray) -> np.ndarray`

Spec §10 requires scoring against the best free public forecast, not only against climatology. The existing `brier_skill_score` hardcodes a climatology baseline.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_metrics.py`:

```python
def test_bss_vs_reference_zero_when_identical():
    from src.evaluation.metrics import brier_skill_score_vs_reference

    probs, labels = _make_preds()
    bss = brier_skill_score_vs_reference(probs, probs, labels)
    assert bss.shape == (4,)
    assert np.allclose(bss, 0.0, atol=1e-9)


def test_bss_vs_reference_positive_when_model_better():
    from src.evaluation.metrics import brier_skill_score_vs_reference

    labels = np.ones((100, 2), dtype=np.float32)
    good = np.full((100, 2), 0.9, dtype=np.float32)
    poor = np.full((100, 2), 0.4, dtype=np.float32)
    bss = brier_skill_score_vs_reference(good, poor, labels)
    assert (bss > 0).all()


def test_bss_vs_reference_negative_when_model_worse():
    from src.evaluation.metrics import brier_skill_score_vs_reference

    labels = np.ones((100, 2), dtype=np.float32)
    good = np.full((100, 2), 0.9, dtype=np.float32)
    poor = np.full((100, 2), 0.4, dtype=np.float32)
    bss = brier_skill_score_vs_reference(poor, good, labels)
    assert (bss < 0).all()


def test_bss_vs_reference_handles_perfect_reference():
    from src.evaluation.metrics import brier_skill_score_vs_reference

    labels = np.ones((20, 1), dtype=np.float32)
    bss = brier_skill_score_vs_reference(labels, labels, labels)
    assert np.isfinite(bss).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/unit/test_metrics.py -k reference -v`
Expected: FAIL with `ImportError: cannot import name 'brier_skill_score_vs_reference'`

- [ ] **Step 3: Write the implementation**

Add to `src/evaluation/metrics.py`, directly after `brier_skill_score`:

```python
def brier_skill_score_vs_reference(
    probs: np.ndarray, reference: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Brier Skill Score against an arbitrary reference forecast, per horizon.

    BSS = 1 - BS_model / BS_reference.

    Unlike ``brier_skill_score``, which compares against climatology, this
    compares against a competing forecast — for CropOS, the free public GFS or
    GraphCast-GFS forecast. "Better than climatology" is a weak claim when a
    farmer can already download a real forecast for nothing; this is the
    number that supports the product claim (spec §10).

    Returns 0.0 for any horizon where the reference is already perfect, since
    no improvement is possible there.
    """
    bs_model = brier_score(probs, labels)
    bs_ref = brier_score(reference, labels)
    return np.where(bs_ref > 1e-9, 1.0 - bs_model / bs_ref, 0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/unit/test_metrics.py -v`
Expected: PASS, including the 4 new tests.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/unit/test_metrics.py
git commit -m "feat(eval): add brier_skill_score_vs_reference

Spec 10 scores against the best free public forecast, not only climatology.
'Better than climatology' is a weak claim when a farmer can download a real
forecast for free."
```

---

## Task 4: Phase 0 benchmark script and gate

**Files:**
- Create: `scripts/phase0_benchmark.py`

**Interfaces:**
- Consumes: `select_run`, `bucket_start`, `gfs_key`, `graphcast_key` (Task 1); `BUCKET_GFS`, `BUCKET_GRAPHCAST`, `fetch_point_values` (Task 2); `brier_skill_score`, `brier_skill_score_vs_reference` (Task 3); `STATION_COORDS` from `src.ingestion.metar`.
- Produces: `data/phase0/forecasts.parquet`, `data/phase0/report.md`, `data/phase0/gate.json`.

**Scope decisions, and why:**
- **00Z runs only, one issuance per day.** 880 days × 2 horizons × 2 models ≈ 3,520 byte-range fetches. Ample statistically (880 days × 16 stations × 2 horizons) and keeps the measurement to hours rather than days.
- **6-hour accumulation window ending at the horizon.** The finest resolution GraphCast-GFS supports natively — it emits only f006/f012/f018/f024/…
- **Window 2024-02-05 → 2025-06-30** (fine-tune window; GraphCast-GFS archive starts 2024-02-05). Validation and test windows are not touched.

- [ ] **Step 1: Write the script**

Create `scripts/phase0_benchmark.py`:

```python
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
        --labels data/raw/era5_thailand.parquet \\
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


def build_labels(
    era5_path: Path, metar_path: Path, valid_times: pd.Series
) -> pd.DataFrame:
    """Build 6-hour-accumulated rain labels from ERA5 and METAR.

    Returns one row per (valid_time, station) with columns:
        era5_rain, metar_rain, metar_valid
    """
    from src.features.dataset import _build_era5_label_df

    era5 = pd.read_parquet(era5_path)
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
    ap.add_argument("--labels", required=True, help="era5_thailand.parquet")
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
```

- [ ] **Step 2: Smoke-test on three days before the full run**

```bash
PYTHONPATH=. python scripts/phase0_benchmark.py \
  --labels data/raw/era5_thailand.parquet \
  --metar data/raw/metar_thai.parquet \
  --out /tmp/phase0_smoke \
  --start 2024-06-01 --end 2024-06-03
```
Expected: completes without error; `forecasts.parquet` has rows for both `gfs` and `graphcast`; report prints with "insufficient" rows (only 3 days, under the n≥100 floor). That is correct behaviour, not a failure.

- [ ] **Step 3: Verify the leakage guard actually bound**

```bash
PYTHONPATH=. python -c "
import pandas as pd
df = pd.read_parquet('/tmp/phase0_smoke/forecasts.parquet')
df['pub'] = df['run'] + pd.Timedelta(hours=4)
assert (df['pub'] <= df['issuance']).all(), 'LEAK: a run was used before publication'
assert (df['lead_h'] >= df['horizon_h']).all(), 'lead shorter than horizon'
print('leakage guard OK:', len(df), 'rows')
print(df.groupby(['model','horizon_h']).lead_h.agg(['min','max']))
"
```
Expected: assertions pass; `lead_h` ≥ `horizon_h` for every row.

- [ ] **Step 4: Commit the script**

```bash
git add scripts/phase0_benchmark.py
git commit -m "feat(phase0): benchmark harness for GFS and GraphCast-GFS

Scores the free public forecasts at the 16 stations at 24h/48h to establish
the bar and gate the grid download. Fetches through select_run() so the
publication-latency rule binds on the benchmark exactly as at inference — a
benchmark that cheats would set the bar too high."
```

- [ ] **Step 5: Run the full benchmark**

```bash
PYTHONPATH=. python scripts/phase0_benchmark.py \
  --labels data/raw/era5_thailand.parquet \
  --metar data/raw/metar_thai.parquet \
  --out data/phase0
```
Expected runtime: 2–5 hours (≈3,500 byte-range fetches). `forecasts.parquet` is checkpointed, so a re-run resumes rather than restarts.

- [ ] **Step 6: Commit results and report the gate**

```bash
git add data/phase0/report.md data/phase0/gate.json
git commit -m "chore(phase0): benchmark results and gate verdict"
```

**STOP HERE.** Report the gate outcome before any further work:

- **PASS** (best calibrated 48 h BSS > 0) → proceed to the main implementation plan; `calibration_a` / `calibration_b` seed the residual head's prior (spec §3.4), and the winning model becomes both prior and benchmark (spec §4.4).
- **FAIL** → do not commission the grid download. Report and revisit the design with the owner, per spec §8.

---

## Self-Review

**Spec coverage (§8 Phase 0):**
- ✓ Pull GFS precip at 24 h/48 h for 16 stations, 2024-02 → present, respecting §4.3 — Task 4 `fetch_forecasts` via `select_run`
- ✓ Pull the same from operational GraphCast-GFS — Task 4, `graphcast_key`
- ✓ Fit 1-D logistic calibration on the fine-tune window only — Task 4 `fit_calibration`, `FIT_START`/`FIT_END`
- ✓ Score both models against both label sources as BSS vs climatology — Task 4 report loop
- ✓ Better of the two becomes the bar; seeds `a`, `b` — Task 4 head-to-head table + `gate.json`
- ✓ Gate with explicit stop — Task 4 Step 6
- ✓ §10 requirement to score against a reference forecast — Task 3
- ✓ §4.3 leakage rule enforced and *verified* — Task 1 tests, Task 4 Step 3

**Deferred to the main plan (not Phase 0):** grid ingestion at scale, the mesh/model rewrite, the ablation, station k-fold, hourly APCP de-accumulation. `bucket_start` is built here because Phase 0 needs it; the differencing rule is not needed until hourly targets exist.

**Placeholder scan:** no TBD/TODO. Every code step contains runnable code. Every test contains assertions.

**Type consistency:** `select_run` returns `(pd.Timestamp, int)` — used as `(run, lead)` in Task 4. `fetch_point_values` returns `dict[str, float]` — consumed as `vals.items()`. `parse_idx` entries carry `start`/`stop`/`descriptor` — used by `find_message` and the Range header. `brier_skill_score_vs_reference(probs, reference, labels)` — argument order matches the Task 4 call `(pc, pg, y)`.

**Lead arithmetic, verified against the live buckets:** at a 00Z issuance the freshest published run is the previous day's 18Z, so a 24 h horizon becomes lead 30 and a 48 h horizon becomes lead 54. Both are 6-hour bucket ends — `bucket_start(30) = 24`, `bucket_start(54) = 48` — and both exist in both buckets with matching descriptors:

```
gfs        f030 -> "APCP:surface:24-30 hour acc fcst"
gfs        f054 -> "APCP:surface:48-54 hour acc fcst"
graphcast  f030 -> "APCP:surface:24-30 hour acc fcst"
graphcast  f054 -> "APCP:surface:48-54 hour acc fcst"
```

So the bucket-end skip in `fetch_forecasts` never fires for the 00Z schedule, and every day yields rows for both models. The skip is retained as a guard for other issuance times, not as an expected path.

Note the semantics this produces: the label window for a "24 h" horizon is the 6 hours *ending* at issuance + 24 h, forecast from a run 30 h earlier. That is a genuine ≥24 h forecast, which is the conservative direction — the measured bar is if anything slightly harder than the model will face.

**GraphCast-GFS precipitation confirmed present** at f024/f030/f048/f054 with the same accumulation convention as GFS, so the head-to-head comparison is like-for-like.
