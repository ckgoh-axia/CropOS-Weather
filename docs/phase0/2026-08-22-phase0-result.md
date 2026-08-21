# Phase 0 — Forecast Benchmark

Window: 2024-05-01 to 2025-06-30  (426 days)
Horizon definitions: "24h" scores the 6-hour accumulation window ending at issuance+24h, forecast from the run published 30h earlier (the freshest run publishable at a 00Z issuance is the previous day's 18Z). "48h" scores the window ending at issuance+48h, from the run published 54h earlier. Both are genuine >=24h / >=48h forecasts.

Forecast rows: 27,072  |  matched to labels: 27,072
ERA5 label windows: 6,832 total, 0 dropped (fewer than all 6 hours present in the accumulation window — a partial sum would silently under-count rain).
ERA5 raw null precipitation rows in the needed window: 0 (checked on the raw frame before label construction — a null-but-present row would otherwise be silently filled to 0.0 mm and reported as complete; a nonzero count here means the run FAILED LOUDLY earlier and this line would not exist, so 0 is the only value that can ever be printed here).

## Fetch coverage (raw, per model — diagnostic only)

This is each model's own fetch success rate. It is NOT what the gate checks — see 'Scored coverage' below, which reflects the gfs∩graphcast matched row set the BSS tables actually score against. A model can read 100% here while its scored coverage is much lower, if the OTHER model has gaps (fix-round-2 NEW-1).

| model | horizon | fetched | expected | fetch coverage |
|---|---|---|---|---|
| gfs | 24h | 6,768 | 6,816 | 99.3% |
| gfs | 48h | 6,768 | 6,816 | 99.3% |
| graphcast | 24h | 6,768 | 6,816 | 99.3% |
| graphcast | 48h | 6,768 | 6,816 | 99.3% |
- Incomplete days (no shard written yet, will retry next run): 3 — 2024-07-26, 2025-04-06, 2025-05-26
- Leakage/bucket-end skips (both models, both horizons): 0
- GFS fetch exceptions: 0 of 852 attempted
- GraphCast-GFS fetch exceptions: 6 of 852 attempted

## Scored against `era5_rain` (>= 1.0mm accumulated over the window)

Fit on the earliest 70% of days in the requested window by issuance date, scored on the held-out latest 30% — never the same rows. `calibration_a`/`calibration_b` in gate.json are refit on the full window once scoring is complete, for seeding the residual head's prior; they are NOT what produced the BSS numbers below. `score-half base rate` is the held-out half's own base rate — if it's 0.000 or 1.000 the cell is degenerate even though the verdict column still reads normally (see gate.json).

| model | horizon | n | base rate | score-half base rate | BSS in-sample | BSS out-of-sample | verdict |
|---|---|---|---|---|---|---|---|
| gfs | 24h | 6,768 (fit 4,736/score 2,032) | 0.186 | 0.156 | +0.1557 | +0.0942 | skilful (BSS>0) |
| gfs | 48h | 6,768 (fit 4,736/score 2,032) | 0.188 | 0.164 | +0.1460 | +0.0809 | skilful (BSS>0) |
| graphcast | 24h | 6,768 (fit 4,736/score 2,032) | 0.186 | 0.156 | +0.3238 | +0.2208 | skilful (BSS>0) |
| graphcast | 48h | 6,768 (fit 4,736/score 2,032) | 0.188 | 0.164 | +0.2982 | +0.2185 | skilful (BSS>0) |

## Scored against `metar_rain` (any rain_event observed in the window)

**Not the same event as `era5_rain`.** `era5_rain` is a >=1.0mm accumulated-depth threshold; `metar_rain` is occurrence-only (was any rain reported at all, from present-weather codes RA/TS/SH), with no intensity information — the Thai ASOS feed's precip_mm field is identically zero across the whole archive upstream and is not usable. Treat this table as a secondary observational sanity check, not a like-for-like comparison with the ERA5 table. It is NOT a gate input.

| model | horizon | n | base rate | score-half base rate | BSS in-sample | BSS out-of-sample | verdict |
|---|---|---|---|---|---|---|---|
| gfs | 24h | 4,590 (fit 3,066/score 1,524) | 0.147 | 0.130 | +0.1049 | +0.0724 | skilful (BSS>0) |
| gfs | 48h | 4,592 (fit 3,068/score 1,524) | 0.148 | 0.135 | +0.0956 | +0.0233 | skilful (BSS>0) |
| graphcast | 24h | 4,590 (fit 3,066/score 1,524) | 0.147 | 0.130 | +0.2486 | +0.1578 | skilful (BSS>0) |
| graphcast | 48h | 4,592 (fit 3,068/score 1,524) | 0.148 | 0.135 | +0.2297 | +0.1520 | skilful (BSS>0) |

## GraphCast-GFS vs raw GFS (the bar)

Positive = GraphCast-GFS beats raw GFS at the same events (scored against `era5_rain`, same fit/score split as above).

| horizon | n (fit/score) | score-half base rate | BSS in-sample | BSS out-of-sample | verdict |
|---|---|---|---|---|---|
| 24h | 4,736/2,032 | 0.156 | +0.1990 | +0.1397 | graphcast better (BSS>0) |
| 48h | 4,736/2,032 | 0.164 | +0.1782 | +0.1496 | graphcast better (BSS>0) |

## Scored coverage (this is what the gate uses)

Computed from the SAME gfs∩graphcast matched row set the BSS tables above score against, not from either model's own raw fetch count. A coverage gap in one model reduces both models' scored coverage here, even when the unaffected model's raw fetch coverage above still reads 100%.

| horizon | matched (gfs∩graphcast) | expected | matched coverage |
|---|---|---|---|
| 24h | 6,768 | 6,816 | 99.3% |
| 48h | 6,768 | 6,816 | 99.3% |

| model | horizon | scored n (era5_rain) | expected | scored coverage |
|---|---|---|---|---|
| gfs | 24h | 6,768 | 6,816 | 99.3% |
| gfs | 48h | 6,768 | 6,816 | 99.3% |
| graphcast | 24h | 6,768 | 6,816 | 99.3% |
| graphcast | 48h | 6,768 | 6,816 | 99.3% |

## Gate

- Best model at 48 h (by out-of-sample BSS vs climatology): **graphcast**
- Out-of-sample BSS at 48 h: **+0.2185** (in-sample: +0.2982)
- Scored coverage at 48 h for graphcast: **99.3%** (raw fetch coverage: 99.3%)
- Thresholds: out-of-sample BSS > 0.02, scored coverage >= 90%
- **PASS — proceed**