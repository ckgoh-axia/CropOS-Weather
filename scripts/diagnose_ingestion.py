#!/usr/bin/env python3
"""
Diagnose ERA5 and METAR ingestion issues without downloading everything.

Run locally:
    python scripts/diagnose_ingestion.py

Takes ~2 minutes. Prints a clear summary of what's broken and why.
"""
from __future__ import annotations

import time
from io import StringIO

import httpx
import pandas as pd

IOWA_STATE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
ERA5_URL = "https://archive-api.open-meteo.com/v1/era5"

# ─────────────────────────────────────────────────────────────────────────────
# METAR diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def check_metar_year(
    station: str, year: int, report_type: int | None, client: httpx.Client
) -> dict:
    params = {
        "station": station,
        "data": "all",
        "year1": str(year), "month1": "1", "day1": "1",
        "year2": str(year), "month2": "12", "day2": "31",
        "tz": "Etc/UTC",
        "format": "comma",
        "latlon": "yes",
        "direct": "yes",
    }
    if report_type is not None:
        params["report_type"] = str(report_type)

    try:
        resp = client.get(IOWA_STATE_URL, params=params, timeout=30.0)
        status = resp.status_code
        if status == 429:
            return {"year": year, "report_type": report_type,
                    "status": 429, "rows": 0, "note": "RATE LIMITED"}
        if status != 200:
            return {"year": year, "report_type": report_type,
                    "status": status, "rows": 0, "note": f"HTTP {status}"}

        lines = [
            ln for ln in resp.text.splitlines()
            if not ln.startswith("#") and ln.strip()
        ]
        if len(lines) <= 1:
            return {"year": year, "report_type": report_type,
                    "status": 200, "rows": 0, "note": "empty response"}

        df = pd.read_csv(StringIO("\n".join(lines)), low_memory=False)
        return {"year": year, "report_type": report_type,
                "status": 200, "rows": len(df), "note": "ok"}
    except Exception as e:
        return {"year": year, "report_type": report_type,
                "status": -1, "rows": 0, "note": str(e)[:80]}


def diagnose_metar():
    print("\n" + "═" * 60)
    print("  METAR DIAGNOSTICS")
    print("═" * 60)

    station = "VTUU"
    print(f"\nStation: {station}")
    print("Testing years 2015–2022 with report_type=1 (current), then no filter")
    print("Adding 2s delay between requests to avoid 429\n")

    row_fmt = "  {year:<6} {status:<8} {rows:>8}  {note}"

    with httpx.Client(timeout=30.0) as client:
        print("  [report_type=1 — what the code currently sends]")
        print(f"  {'Year':<6} {'Status':<8} {'Rows':>8}  Note")
        print(f"  {'-'*50}")
        for year in range(2015, 2023):
            result = check_metar_year(station, year, report_type=1, client=client)
            print(row_fmt.format(**result))
            time.sleep(2.0)

        print()
        print("  [no report_type filter — all observation types]")
        print(f"  {'Year':<6} {'Status':<8} {'Rows':>8}  Note")
        print(f"  {'-'*50}")
        for year in range(2015, 2023):
            result = check_metar_year(station, year, report_type=None, client=client)
            print(row_fmt.format(**result))
            time.sleep(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# ERA5 diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_era5():
    print("\n" + "═" * 60)
    print("  ERA5 DIAGNOSTICS")
    print("═" * 60)

    import numpy as np

    lats_all = list(np.arange(5.5, 20.5, 0.25))
    lons_all = list(np.arange(97.5, 105.7, 0.25))
    all_pairs = [(lat, lon) for lat in lats_all for lon in lons_all]

    test_batches = [
        ("Batch 1  (should work)", all_pairs[0:10]),
        ("Batch 3  (where it stopped)", all_pairs[20:30]),
        ("Batch 10 (mid-grid)", all_pairs[90:100]),
    ]

    print()
    for i, (label, batch) in enumerate(test_batches):
        lats = [p[0] for p in batch]
        lons = [p[1] for p in batch]
        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": "2015-01-01",
            "end_date": "2015-01-07",
            "hourly": ["precipitation", "temperature_2m"],
            "timezone": "UTC",
        }
        try:
            resp = httpx.get(ERA5_URL, params=params, timeout=30.0)
            print(f"  {label}: HTTP {resp.status_code}", end="")
            if resp.status_code == 200:
                data = resp.json()
                n = len(data) if isinstance(data, list) else 1
                print(f" → {n} locations returned  ✓")
            else:
                body = resp.text[:200].replace("\n", " ")
                print(f" → {body}")
        except Exception as e:
            print(f"  {label}: ERROR — {e}")

        if i < len(test_batches) - 1:
            print("  (waiting 15s before next batch...)")
            time.sleep(15.0)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("CropOS ingestion diagnostics")
    print("Testing METAR and ERA5 with minimal requests to identify failures")

    diagnose_metar()
    diagnose_era5()

    print("\n" + "═" * 60)
    print("  Done. Paste the full output to diagnose.")
    print("═" * 60)


if __name__ == "__main__":
    main()
