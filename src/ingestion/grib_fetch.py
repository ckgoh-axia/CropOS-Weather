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

    start, stop = msg["start"], msg["stop"]
    end = "" if stop is None else str(stop - 1)
    headers = {"Range": f"bytes={start}-{end}"}
    resp = requests.get(f"{bucket}/{key}", headers=headers, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()

    # A proxy or CDN that strips the Range header returns 200 with the
    # WHOLE object instead of 206 with just the requested bytes. Decoding
    # that as if it were the single requested GRIB message would silently
    # feed the wrong message's bytes to cfgrib — accept only a real partial
    # response, and only if it's exactly the length we asked for.
    if resp.status_code != 206:
        raise ValueError(
            f"Range request not honoured for {bucket}/{key}: expected HTTP "
            f"206 Partial Content, got {resp.status_code}. A proxy may be "
            "stripping the Range header."
        )
    if stop is not None:
        expected_len = stop - start
        if len(resp.content) != expected_len:
            raise ValueError(
                f"Range response for {bucket}/{key} has {len(resp.content)} "
                f"bytes, expected {expected_len} (bytes={start}-{end}). A "
                "proxy may have altered the response."
            )

    return _decode_points(resp.content, coords)
