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


def _msg_len() -> int:
    entries = parse_idx(_IDX)
    msg = find_message(entries, "APCP:surface:18-24 hour acc fcst")
    return msg["stop"] - msg["start"]


@patch("src.ingestion.grib_fetch.requests.get")
def test_fetch_point_values_requests_only_the_needed_bytes(mock_get):
    from src.ingestion.grib_fetch import fetch_point_values

    idx_response = MagicMock(status_code=200, text=_IDX)
    # Content length must match the requested byte range exactly, or the
    # new honoured-Range check (below) rejects it.
    grib_response = MagicMock(status_code=206, content=b"x" * _msg_len())
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


@patch("src.ingestion.grib_fetch.requests.get")
def test_fetch_point_values_rejects_unhonoured_range(mock_get):
    """A proxy that strips Range returns 200 + the whole object instead of
    206 + just the requested bytes. Silently decoding that would feed the
    wrong message to cfgrib — must raise instead."""
    from src.ingestion.grib_fetch import fetch_point_values

    idx_response = MagicMock(status_code=200, text=_IDX)
    grib_response = MagicMock(status_code=200, content=b"x" * _msg_len())
    mock_get.side_effect = [idx_response, grib_response]

    with pytest.raises(ValueError, match="206"):
        fetch_point_values(
            BUCKET_GFS,
            "gfs.20260801/00/atmos/gfs.t00z.pgrb2.0p25.f024",
            "APCP:surface:18-24 hour acc fcst",
            {"VTUU": (15.25, 104.87)},
        )


@patch("src.ingestion.grib_fetch.requests.get")
def test_fetch_point_values_rejects_wrong_length_range(mock_get):
    """206 but a body of the wrong length means something altered the
    response in transit — must raise rather than decode a truncated or
    padded message."""
    from src.ingestion.grib_fetch import fetch_point_values

    idx_response = MagicMock(status_code=200, text=_IDX)
    grib_response = MagicMock(status_code=206, content=b"x" * (_msg_len() - 1))
    mock_get.side_effect = [idx_response, grib_response]

    with pytest.raises(ValueError, match="bytes"):
        fetch_point_values(
            BUCKET_GFS,
            "gfs.20260801/00/atmos/gfs.t00z.pgrb2.0p25.f024",
            "APCP:surface:18-24 hour acc fcst",
            {"VTUU": (15.25, 104.87)},
        )
