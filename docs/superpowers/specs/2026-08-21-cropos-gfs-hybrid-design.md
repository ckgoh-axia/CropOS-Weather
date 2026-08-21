# CropOS GFS-Hybrid Redesign — Design Spec

**Date:** 2026-08-21
**Revision:** 2 — GFS-direct ingestion; leakage controls; station k-fold ablation
**Status:** Approved for planning
**Supersedes:** `2026-08-14-nwp-forecast-integration.md` (do not execute; see §11)

---

## 1. Purpose

Two goals, in priority order:

1. **Business case.** Produce a defensible number for what a local weather
   station is worth to farm-level rain forecasting in Thailand. The 16 Thai
   airport METAR stations stand in for the cheap farm sensors CropOS intends
   to deploy.
2. **Model skill.** Beat the best free public forecast at the same locations
   at ~24 h and ~48 h lead.

Goal 1 is the deliverable; goal 2 is what makes it worth measuring.

**Predictions are hourly-resolved.** Station inputs are hourly; grid inputs
are 6-hourly (§3.3). Daily aggregates are derived from hourly outputs, never
modelled directly — sub-daily timing is required by the spray/washoff
decisions in `configs/evaluation.yaml`.

---

## 2. Why the current model scores BSS −0.6 to −0.8

Four causes, in descending order of estimated contribution. The superseded
handover identifies only the third.

### 2.1 Receptive field truncated to ~100 km

`configs/model.yaml: edge_radius_km: 100` and
`CropOSDataset(era5_node_radius_km=100.0)` clip the atmospheric grid to a
100 km halo around the stations — 100–200 grid points per the
`src/features/dataset.py` module docstring. Message passing then runs 4
rounds over 16 station nodes with `metar_to_metar_k: 4`.

At 48 h, advection at ~10 m s⁻¹ carries systems ~1,700 km. The weather that
determines rain over Ubon Ratchathani in two days is outside the graph. No
feature engineering recovers information the graph cannot represent.

Both reference papers solve exactly this and CropOS adopts neither solution:
Keisler (2022) and Lam et al. (2023) process on an icosahedral mesh —
GraphCast's *multi-mesh* — so information crosses long distances within a
fixed number of message-passing rounds.

### 2.2 No forecast information reaches the model

Inputs are analyses and observations at times ≤ t; labels are rain at
t+12…t+48. The model must extrapolate 48 h from surface data with no forecast
product and no dynamics. This is the hardest possible framing of the task.

### 2.3 The NWP pipeline exists but is unused

`src/features/dataset.py` downloads `nwp_features.parquet` and discards it
("NOT used in training"). Correct as a finding; the superseded fix does not
work (§11.1).

### 2.4 Architectural details

- **No LayerNorm anywhere.** `_RelPosBipartiteConv` / `_RelPosSelfConv`
  docstrings claim residuals make training "stable without layer
  normalisation". GraphCast normalises every MLP. With `history_steps: 24`
  the station projection input is 1,152-dim and unnormalised.
- **Absolute prediction, not residual.** Both papers predict a delta from
  current state; CropOS predicts absolute probability, so it has no skill
  floor.
- **`pos_weight` tuning history** (16.5 → 3.0 → 2.5 across `7821806`,
  `e976f52`) is calibration-chasing on a model with no signal — a symptom,
  not a cause.

### 2.5 Deviation from the cited papers

`src/models/gnn.py` cites **arXiv:2410.12938**, not Keisler or GraphCast; the
handover's "theoretical basis" is a retrofit. Against Keisler/GraphCast the
current model deviates on every structural axis: no mesh, no autoregression,
no vertical levels, no residual prediction, no normalisation. The claim that
`local_mp_steps: 4` "corresponds to M=4 in the paper" does not describe
GraphCast, which uses 16 processor layers on a multi-mesh.

**Decision:** adopt Keisler's *encode–process–decode with a multi-scale mesh*
(§3). Do **not** adopt autoregressive state prediction — §5.4 explains why the
correction framing is both cheaper and more production-viable here.

---

## 3. Target architecture

### 3.1 Node types

| Type | Count | Source | Cadence | Live at inference? |
|---|---|---|---|---|
| `grid` | 1,600 (v1) | **GFS GRIB2, NOAA AWS**, 22 vars incl. pressure levels | 6-hourly | **Yes** |
| `station` | 16 (extensible) | METAR observations, 9 vars | hourly | **Yes** |
| `farm` | arbitrary | prediction targets, any lat/lon | hourly | n/a |

**ERA5 is not a model input.** It survives only as a training label (§4.6).

`era5` is renamed `grid` and `metar` renamed `station` throughout — the node
type is no longer tied to a data source. Breaking rename, done now.

### 3.2 Encode–process–decode

```
encode:   grid nodes ──radius edges──▶ mesh nodes
process:  8 rounds of message passing on a MULTI-SCALE mesh
decode:   mesh ──▶ station nodes ──▶ farm nodes ──▶ residual head
```

**Multi-scale mesh.** Rather than build an icosahedron for a regional domain,
edges are constructed on one mesh node set at three ranges: nearest-neighbour
(~1°), medium (~4°), long (~8°). This reproduces the property that makes
GraphCast's multi-mesh work — a message crosses the domain in ~4 hops instead
of ~40 — without global icosahedral projection machinery.

**This is not optional.** Enlarging the domain without long-range edges adds
cost and no capability. The two changes land together or neither is worth
making.

**Processor depth:** 8 rounds (Keisler 6, GraphCast 16).
**LayerNorm in every MLP**, both conv classes and all heads.

### 3.3 Domain, resolution and cadence

**v1: 1.0° across 10 °S–30 °N, 85–125 °E = 1,600 nodes.**

Roughly 10× the current receptive field, reaching ~1,700 km upstream — the
advection distance at 48 h. Rationale in §5.3.

A 0.5°-over-Thailand refinement (~2,100 nodes total) is prepared but gated on
validation evidence. Note this is a *resolution* decision, not a *domain*
decision: the two have different cost profiles (§5.3).

**Cadence: grid 6-hourly, stations hourly.** GFS runs 4×/day, and both
reference papers operate on 6-hourly timesteps, so 6-hourly grid state is
paper-faithful and matches the data's native cadence. Station observations
remain hourly. Predictions are hourly-resolved: the hourly station branch and
the interpolated/held grid state jointly drive an hourly decoder.

### 3.4 The GFS residual head

For horizon *h*, at issuance time *T*:

```
prior_h   = a · log1p(gfs_precip_mm @ T+h) + b      # a, b learned scalars
Δ_h       = MLP([farm_embedding, gfs_h, horizon_embedding])
logit_h   = prior_h + Δ_h
p_h       = sigmoid(logit_h)
```

`gfs_h` is the **full 22-variable GFS field** valid at T+h, taken from a run
whose publication time is ≤ T (§4.3). Native GRIB carries every variable at
every lead, so the surface-only restriction that applies to Open-Meteo's
`_previous_dayN` API does not apply here.

Δ's final layer is zero-initialised, so **the model begins training at the
prior's skill and learns only the correction.** This is Keisler's
predict-the-residual property and supplies a skill floor the current
architecture lacks entirely.

`a` and `b` are initialised from the Phase 0 logistic fit (§8).

**Prior source is a Phase 0 decision:** raw GFS or operational GraphCast-GFS
(§4.4). Whichever scores better becomes both the prior and the benchmark.

### 3.5 Why GFS inputs rather than ERA5 — and what the papers do

| | Trains on | Serves on | Production-viable as published? |
|---|---|---|---|
| Keisler 2022 | ERA5 (Copernicus) | — | **No.** ERA5 has multi-day latency. |
| GraphCast 2023 | ERA5 | HRES operational analysis | Yes, but only by explicitly bridging the ERA5→HRES distribution shift |
| **CropOS** | **GFS operational** | **GFS operational** | **Yes — same source both sides, zero shift** |

Both papers train on reanalysis they cannot use operationally; GraphCast
spends real effort at inference bridging that gap. CropOS avoids the problem
by training and serving on one operational source — possible precisely
*because* it is a correction model rather than a from-scratch forecaster.

Measured: ERA5 via Open-Meteo `archive-api` has ~1 day latency, so an
ERA5-fed model would run on a stale state and on a distribution it was not
trained for. GFS analysis fields are available live, carry full vertical
structure, and are the honest baseline for the ablation (§5): arm A must
represent what a farmer can actually obtain today.

### 3.6 Farm nodes decoupled from station nodes

`CropOSDataset` currently builds `farm_node_list` from `station_coords`,
pinning farm nodes to the 16 airports. Farm nodes must accept an arbitrary
`(lat, lon)` list independent of the station set, so a sensor can be sited
anywhere without retraining. Default remains the 16 airport locations.

---

## 4. Data — measured availability

All figures verified by direct probe on 2026-08-21, not taken from
documentation.

### 4.1 Primary source: NOAA GFS on AWS (**chosen**)

```
s3://noaa-gfs-bdp-pds/
  gfs.20210101/                                       ← archive starts 2021-01-01
  gfs.20210301/00/gfs.t00z.pgrb2.0p25.f024            ← run 00Z, lead hour 024
  gfs.20260801/00/atmos/gfs.t00z.pgrb2.0p25.f024.idx  ← 743 messages, byte-range subsettable
```

- **Public domain (NOAA NODD). Free. No commercial restriction. No rate limit.**
- Native 0.25° global, all variables, all pressure levels, all lead hours.
- Addressed by **explicit run and explicit lead hour**.

That last property is the decisive one: it eliminates an entire class of
leakage bug rather than mitigating it (§4.3). Note the path layout changes —
`atmos/` appears in later years but not in 2021 — so the fetcher must handle
both.

**Volume:** ~188 GB for 6-hourly analysis 2021-01 → 2026-06; ~564 GB
including f024/f048. Free S3 egress. After regional crop: **1.1 GB at 1.0°,
18 GB at 0.25°.**

### 4.2 Open-Meteo — retained only as a fallback

Standard $29/mo (1M calls/mo), Professional $99/mo (5M). Free tier is 10k/day
and **non-commercial only**, so any production use requires a paid plan.
Backfill under fractional counting would be ~480k calls at 1,600 nodes.

Cost is trivial, but GFS-direct is preferred for the leakage and variable-
coverage reasons above. Open-Meteo remains a viable fallback for METAR-
adjacent point queries and for rapid prototyping.

### 4.3 Leakage control — the binding constraint

**Rule: every input field must come from a model run whose *publication* time
is ≤ issuance time T.** Initialisation time is not sufficient; GFS publishes
~4 h after initialisation.

With GFS-direct this is enforced structurally: select run `r` such that
`r + 4h ≤ T`, then read lead hour `h` from that run. Both the grid state and
the residual prior come from the **same run**, so the guarantee is one
condition, checked once.

Two leaks this replaces, both found in revision 1 of this spec and both
present in the superseded plan's data source:

- **Open-Meteo `historical-forecast-api`** stitches the *first hours* of
  successive runs. The value at time *t* comes from a run initialised at
  t−0…t−6 and published up to t+4. **Roughly half of all hours are not yet
  published at their own valid time.** Using it at times ≤ t is therefore
  *not* safe, contrary to what revision 1 of this spec asserted.
- **Open-Meteo `_previous_dayN`** was verified by exact-match probe
  (d = 0.00) to stitch runs initialised **24–29 h before valid time**:

  | Valid hour | Source run | Implied lead |
  |---|---|---|
  | 00Z | V−1 00Z | 24 h |
  | 03Z | V−1 00Z | 27 h |
  | 06Z | V−1 06Z | 24 h |
  | 12Z | V−1 12Z | 24 h |

  A genuine ≥24 h forecast, but the latest contributing run publishes at
  V−20. Treating it as issued at V−24 leaks by up to 4 h.

**Additional leakage controls, all mandatory:**

- **Split embargo ≥ 72 h** at every boundary (max history + max horizon), so
  no training sample carries a label inside the next split.
- **Scalers fitted on the training split only, and re-fitted per ablation
  arm.** Sharing arm B's scalers with arm A leaks val/test statistics into
  the baseline and corrupts the headline B−A number.
- **Rolling derived features** (`nwp_precip_24h_sum`, `nwp_dp_3h`) computed
  causally, documented at the definition site.
- **No `np.nan_to_num(..., nan=0.0)` without an accompanying validity mask.**
  Audit every existing call (§9).

### 4.4 Operational GraphCast-GFS — benchmark and candidate prior

```
s3://noaa-nws-graphcastgfs-pds/
  graphcastgfs.20240205/   ← directory tree exists from 2024-02-05
  aigfs.20260416/          ← newer EAGLE model, from 2026-04
```

NOAA runs GraphCast operationally, free, from GFS initial conditions.

1. **It is a candidate prior** — correcting a model that already beats GFS.
2. **It is a harder benchmark.** If GraphCast-GFS beats raw GFS at these 16
   stations, that is the bar. Measuring only against raw GFS would be a straw
   man. Phase 0 measures both (§8).

**"Archive exists" is not "byte-range fetchable" — these are two different
dates.** The `graphcastgfs.YYYYMMDD/` directories and GRIB data files exist
from 2024-02-05 and return HTTP 200 throughout. But `fetch_point_values`
locates a message's byte range via the `.idx` sidecar file, and that sidecar
is not published until later: verified counts in `forecasts_13_levels/` —
`20240424/00` has 0 `.idx` files, `20240428/00` has 40, `20240501/00` has 40,
`20250630/18` has 65. The archive is therefore only usable by this pipeline
from **2024-05-01**, not 2024-02-05. As a *prior* it constrains the
fine-tune window to start no earlier than 2024-05-01; as a *benchmark* it
applies to validation and test regardless.

### 4.5 What is not obtainable

- **ERA5 pressure levels from Open-Meteo:** all null at every date and every
  `models=` value tried. Would require a Copernicus CDS account and a gridded
  download. Recorded so the option is not rediscovered.
- **GFS before 2021-01** on the NODD bucket. Deeper GFS archives exist
  elsewhere (e.g. NSF NCAR GDEX d084001) and are out of scope for v1.

### 4.6 Labels

- **Training label:** ERA5 `precipitation` at the nearest grid point to each
  farm node, thresholded at 1.0 mm — unchanged. Measured (not assumed) dense
  and gap-free: 20,197,980 rows over 2024-05-01..2025-06-30 with 0 nulls and
  0 NaNs, measured 2026-08-22 against `era5_recent.parquet`. This was
  previously an unverified assumption in this spec; the Phase 0 harness now
  checks the raw frame for null-but-present rows before they can reach label
  construction and refuses to proceed if any are found, rather than trusting
  this measurement to still hold on a future re-run against different data.
- **Reporting label:** ~~METAR `precip_mm` with a validity mask~~ — **not
  usable.** The Thai ASOS feed the METAR ingestion reads has no `p01i`
  column, so `src/ingestion/metar.py:141`'s
  `pd.to_numeric(df.get("p01i", 0), ...)` falls back to its literal `0`
  default for every row: `precip_mm` is identically zero across the entire
  archive (verified independently, 1,034,434 rows, min/max/sum all 0.0).
  The Phase 0 harness instead uses `rain_event` — an occurrence flag derived
  from present-weather codes (RA/TS/SH), with no accumulated depth. **This
  is a different event from the training label's `era5_rain` ≥1.0 mm
  accumulated-depth threshold — they are not directly comparable, and the
  harness's report keeps that warning prominent next to every `metar_rain`
  table.** Headline BSS is reported against `era5_rain`; the `metar_rain`
  table is a secondary observational sanity check only, never a gate input.

Labels are needed only offline, so ERA5's latency is irrelevant to production.
**No production input depends on ERA5.**

**Target mismatch must be corrected explicitly.** ERA5 grid-cell-mean accumulation
≥1 mm and METAR `rain_event` occurrence are different events. A recalibration
fitted on the fine-tune window only maps one to the other; without it the
reported number carries a systematic offset.

`dataset.py` already warns when METAR `zero_frac > 0.60`. Masked METAR
evaluation must report coverage alongside every metric, so high- and
low-coverage stations are never averaged silently.

### 4.7 Splits

| Split | Window | ≈ 6-hourly steps | Residual head |
|---|---|---|---|
| Pretrain | 2021-01 → 2024-01 | ~4.4k | inactive |
| Fine-tune | 2024-02 → 2025-06 | ~2.1k | active |
| Validation | 2025-07 → 2025-12 | ~0.7k | active |
| Test | 2026-01 → 2026-06 | ~0.7k | active |

**Caveat on "Residual head: active" for the first ~3 months of Fine-tune
(2024-02-05 → 2024-04-30):** GraphCast-GFS's `.idx` byte-range sidecars —
required to fetch it as the residual head's prior — are not actually
fetchable until 2024-05-01, even though the `graphcastgfs.YYYYMMDD/`
directories exist from 2024-02-05 (§4.4 — "archive exists" and "byte-range
fetchable" are different dates). The Fine-tune window's declared start
(2024-02-05) is kept as the leakage-protection boundary (it is what
`FIT_START` in `scripts/phase0_benchmark.py` bounds `--start`/`--end`
against, and matches the plan's stated calibration window), but a residual
head trained or evaluated against a GraphCast prior in
2024-02-05..2024-04-30 has no real prior to draw on for those dates —
implementation must either fall back to climatology there or treat that
sub-window as pretrain-like. `GRAPHCAST_IDX_FROM` (2024-05-01) is the
separate, practical constant for "GraphCast is actually fetchable from
here" — see the code comment.

Hourly station samples give ~6× these counts at the decoder. ≥72 h embargo at
every boundary (§4.3). Test is touched once, at the end.

---

## 5. Expectations, and what limits them

### 5.1 Realistic gain

The model cannot out-forecast GFS dynamically — GFS runs a full physical
integration. Its value is exactly two things: **regime-conditioned bias
correction**, and **METAR observations from after the GFS run's
initialisation**. The second decays quickly with lead.

**Expect a meaningful gain at ~24 h and a small one at ~48 h.** Given the
`pos_weight` tuning history (§2.4), this must be agreed before training
starts rather than discovered after.

### 5.2 Graceful degradation is a design property

If the station feed or the mesh fails, emitting `prior_h` alone yields
**calibrated raw GFS**. The system degrades to "as good as the free public
forecast", not to garbage. This is a stated requirement with a tested
fallback path, not an accident.

### 5.3 Effective sample size is the binding constraint

2021-01 → 2025-06 is ~6.5k 6-hourly grid steps. Rain occurrence decorrelates
in ~6–12 h and the 16 stations are spatially correlated, so **effective
independent sample size is order 10³–10⁴, not 10⁵.**

**Domain size is not the lever for this.** GNN parameters are shared across
nodes: adding grid nodes increases compute and input information, but not
parameter count. Overfitting risk lives in hidden width, processor depth and
the head — not in how many grid points feed them. An earlier revision of this
spec conflated the two and recommended shrinking the domain on statistical
grounds; that reasoning was wrong.

RunPod multi-GPU removes the compute constraint and GFS-direct removes the
cost constraint, so **the domain is set by meteorology (§3.3), and capacity is
controlled independently** via hidden width, depth, dropout, the
zero-initialised residual head, and early stopping.

What effective sample size *does* constrain: hidden width and processor depth
should start modest (hidden ≤ 256, 8 rounds) and grow only on evidence of
underfitting. It also means validation must be judged on the station k-fold
(§6.1), where the ~10³ figure is the honest denominator for confidence
intervals.

Resolution — unlike domain — is a genuine trade: 0.25° over Thailand carries
16× the grid nodes of 1.0° for the same area, and 18 GB vs 1.1 GB. Gated on
validation evidence.

### 5.4 Why not Keisler-faithful autoregression

It requires pressure-level ERA5, unavailable from the current stack (§4.5),
and would need to out-forecast GFS from scratch on ~10³ effective samples.
The correction framing needs far less data for the same operational benefit
and, per §3.5, is more production-viable than either published paper.

---

## 6. The ablation — primary deliverable

Arms, identical in every respect except the station branch:

| Arm | Station nodes | Measures |
|---|---|---|
| **A** | none | no local sensors at all |
| **B** | all available | own sensor + neighbour network |
| **C** | all but the target station | neighbour network alone |

Derived: **C − A** = value of a neighbour network; **B − C** = value of a
sensor at your own site; **B − A** = total. These price two distinct products.

### 6.1 Spatial hold-out is mandatory

Evaluating on stations the model trained at answers "what if an existing
sensor fails" — an easier question that **overstates the value of a new
deployment**, because farm-node position is an input and the model memorises
site climatology.

Your actual question is *"if I install a sensor at a new farm, does it
help?"* — a site never seen in training.

**Therefore: k-fold over stations, 4 folds of 4 held out entirely from
training.** All arms are scored on held-out stations only. This is the number
that survives contact with a customer.

### 6.2 Validity constraints — requirements, not guidance

1. A and B share grid data, splits, folds, seed, optimiser, schedule and
   epoch budget. The **only** difference is the station branch.
2. Arm C is produced by masking the target station at evaluation on the
   trained arm-B model — not by training 16 models.
3. Arm C is legitimate **only because** arm B trains with station DropNode,
   making "this station is absent" in-distribution.
4. Scalers re-fitted per arm and per fold (§4.3).

### 6.3 `metar_dropout`, corrected and repurposed

`metar_dropout: 0.4` currently drops station embeddings after projection. Two
changes:

1. **It applies to the station branch only, never to the GFS branch.** Under
   the superseded plan's design (§11.1), 40 % of samples would have had their
   GFS features zeroed — regularising away the signal being added.
2. Its purpose is now to make arm C valid, documented at the definition site.
   Its value is an experimental parameter: too high and arm B understates the
   value of stations; too low and arm C is out-of-distribution. Tune on
   validation; report the chosen value.

---

## 7. Corrections carried over from the superseded plan

- **`nwp_min_start` is wrong.** `configs/data.yaml` sets `2016-01-01`;
  `src/ingestion/nwp_baseline.py` documents the same. Probes return 0/24
  non-null at 2016/2018/2020/2021-03 and 24/24 at 2021-04. Under the
  superseded plan this would have produced five years of null rows, converted
  to zeros by `np.nan_to_num`, training silently on fabricated data. **Fix
  regardless of what else is built.**
- **`history_steps` mismatch.** `configs/model.yaml` sets 24; the handover
  assumes 6 (`metar_in = 288 = 48×6`). With 24, `train.py` computes
  `48 × 24 = 1152`. The specified smoke test could not pass. Set explicitly
  and assert in a test.
- **Static features projected once.** The 11 static columns are constant in
  time; stacking them ×`history_steps` produces 264 duplicated dimensions and
  the one-hot columns hit `FeatureScaler`'s 1e-8 std floor. Project once,
  concatenate after temporal stacking.

---

## 8. Phase 0 — measure the bar before building

Before any architecture work:

1. Pull GFS `precipitation` at leads 24 h and 48 h for the 16 stations, over
   the fine-tune window only — **2024-02-05 → 2025-06-30**, not into
   validation or test — from NOAA AWS, respecting §4.3. (In practice the
   GraphCast-GFS half of this cannot start before 2024-05-01 — see §4.4 —
   so a real run's usable range is 2024-05-01 → 2025-06-30; the harness
   refuses `--start`/`--end` outside 2024-02-05..2025-06-30 entirely unless
   overridden with `--allow-outside-fit-window`, since the calibration fit
   must never touch validation/test data. This is a correction from an
   earlier revision of this section, which asked for 2024-02 → 2026-06 —
   an interval that reaches straight through validation and test. The
   implemented refusal is correct; this text was wrong.)
2. Pull the same from operational GraphCast-GFS (§4.4).
3. Fit 1-D logistic calibrations from precip mm to rain probability on the
   fine-tune window only.
4. Score raw and calibrated GFS **and** GraphCast-GFS at both horizons,
   against **both** label sources, as BSS vs climatology.

**The better of the two is the bar**, and it seeds `a`, `b` in §3.4 and
selects the prior.

**Gate (as implemented):** the best calibrated prior's **out-of-sample**
Brier Skill Score at 48 h (fit on the earliest 70% of fine-tune-window days,
scored on the held-out latest 30% — never the same rows) must exceed
**0.02**, AND the **scored coverage** at 48 h for that model (rows actually
matched between GFS and GraphCast-GFS and scored against `era5_rain`,
divided by expected rows) must be **≥ 90%**. Both conditions are required;
"near zero or negative" from an earlier revision of this section undersold
the coverage requirement entirely and described the BSS threshold as an
informal boundary rather than the two-part pass/fail rule now enforced by
`scripts/phase0_benchmark.py`'s `GATE_MIN_BSS_48H` / `GATE_MIN_COVERAGE`. If
either condition fails, there is too little to correct (or too little of
the run actually landed to trust the number), and the design must be
revisited before the grid download is commissioned. Stop and report rather
than proceeding.

---

## 9. Ingestion

- Fetch GFS GRIB2 from `s3://noaa-gfs-bdp-pds` by **explicit run and lead
  hour**, using `.idx` byte-range requests to pull only required messages
  (743 per file; we need ~30). Handle both `gfs.YYYYMMDD/HH/` and
  `gfs.YYYYMMDD/HH/atmos/` layouts.
- Extend the per-station checkpoint pattern in
  `nwp_baseline.py::fetch_all_stations` to **(run, lead) tiles**; partial
  downloads resume rather than restart.
- Crop to the domain, write parquet/zarr keyed by `(run, lead, lat, lon)`.
  Keeping run and lead in the key is what makes §4.3 auditable after the fact.
- **Coverage verification is mandatory.** After every fetch, assert non-null
  fraction per variable per tile against an explicit threshold and **fail
  loudly**. The `nwp_min_start` bug (§7) would have been caught immediately by
  such a check; `np.nan_to_num(…, nan=0.0)` is precisely what let it pass
  silently.

---

## 10. Evaluation protocol

- **Primary:** BSS at 24 h and 48 h against the **best calibrated free public
  forecast** (raw GFS or GraphCast-GFS, per §8) — not only against
  climatology. "Better than what a farmer can already get free" is the claim
  that matters.
- **Secondary:** reliability diagrams, POD/FAR/CSI at the operational
  threshold, and the agricultural decision categories in
  `src/evaluation/agri_classifier.py`, including sub-daily spray/washoff
  timing.
- **Ablation:** §6, on held-out stations, with bootstrap confidence intervals,
  per horizon and per station.
- **Coverage:** METAR label coverage reported next to every METAR-scored
  metric.
- Test window evaluated once.

---

## 11. Disposition of `2026-08-14-nwp-forecast-integration.md`

**Do not execute.** Retain for reference.

### 11.1 Why

The plan indexes NWP at `past_ts = ts - step * 1h` (plan lines 454–520) while
labels sit at t+12…t+48. **The GFS forecast valid at the target time is never
read.** A GFS-correction model that never sees the forecast it corrects is not
a correction model. Expected effect: some gain at 12 h from better
analysis-time variables, approximately none at 48 h.

Combined with §7 (five years of nulls silently zeroed), §6.3 (DropNode
destroying 40 % of the new signal), and the `history_steps` / static-stacking
defects, the plan would have produced a model that is larger, slower, trained
partly on fabricated data, and no more skilful.

### 11.2 What it got right, and is carried forward

- The diagnosis that NWP data is downloaded and discarded.
- `prepare_nwp_features()` and its derived-feature set.
- The per-station checkpointing pattern, extended to (run, lead) tiles (§9).
- Backward-compatible optional-argument style for dataset construction.

---

## 12. Out of scope

- Autoregressive state prediction (§5.4).
- Copernicus CDS ingestion / ERA5 pressure levels (§4.5).
- GFS before 2021-01 (§4.5).
- Ensemble / probabilistic GFS (GEFS members).
- 0.25° resolution refinement over Thailand — prepared, gated on §5.3 evidence.
- Re-tuning `pos_weight` before the residual head exists (§2.4).

---

## 13. Open questions for implementation

1. **Mesh construction:** mesh nodes coincident with grid nodes, or a
   separate coarser set? Coincident is simpler; separate is closer to Keisler.
   **Default: coincident for v1.**
2. **Prior source:** raw GFS or GraphCast-GFS — decided by Phase 0 (§8).
3. **`metar_dropout` value:** experimental parameter, tuned on validation
   (§6.3).
4. **Hourly decoder against 6-hourly grid state:** hold the grid embedding
   constant across the 6 h window, or interpolate between adjacent states?
   **Default: hold; revisit if validation shows timing errors clustered at
   window boundaries.**
5. **Arm A hyperparameters:** own tuning, or share arm B's for comparability?
   **Default: share, and document it** — this is the most likely challenge to
   the ablation's validity and therefore to the business case.
