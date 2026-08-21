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
