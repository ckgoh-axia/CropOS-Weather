# CropOS GFS-Hybrid Redesign — Design Spec

**Date:** 2026-08-21
**Status:** Approved for planning
**Supersedes:** `2026-08-14-nwp-forecast-integration.md` (do not execute that plan; see §9)

---

## 1. Purpose

Two goals, in priority order:

1. **Business case.** Measure what a local weather station is worth for
   farm-level rain forecasting in Thailand, as a defensible number. Thai
   airport METAR stations stand in for the cheap farm stations CropOS
   intends to deploy.
2. **Model skill.** Produce 24 h and 48 h rain-occurrence forecasts that beat
   the free public GFS forecast at the same locations.

Goal 1 is the deliverable. Goal 2 is what makes goal 1 worth measuring. The
architecture is chosen to serve both, and where they conflict, goal 1 wins.

Horizons are **24 h and 48 h only**. 12 h and 36 h are dropped (§4.3).

---

## 2. Why the current model scores BSS −0.6 to −0.8

Four causes, in descending order of estimated contribution. The existing
handover identifies only the third.

### 2.1 Receptive field truncated to ~100 km

`configs/model.yaml: edge_radius_km: 100` and
`CropOSDataset(era5_node_radius_km=100.0)` clip the atmospheric grid to a
100 km halo around the 16 stations — 100–200 grid points per
`src/features/dataset.py` module docstring. Message passing then runs 4
rounds over 16 station nodes with `metar_to_metar_k: 4`.

At 48 h lead, advection at ~10 m s⁻¹ carries systems ~1,700 km. The weather
that determines rain over Ubon Ratchathani in two days is currently outside
the graph entirely. No feature engineering can recover information the graph
cannot represent.

Both reference papers solve exactly this problem and CropOS does not adopt
either solution: Keisler (2022) and Lam et al. (2023) both process on an
icosahedral mesh — GraphCast's *multi-mesh* — specifically so that
information crosses long distances within a fixed number of message-passing
rounds.

### 2.2 No forecast information reaches the model

Inputs are ERA5 analysis and METAR observations at times ≤ t. Labels are rain
at t+12…t+48. The model is asked to extrapolate 48 h from surface
observations with no forecast product and no atmospheric dynamics. This is
the single hardest possible framing of the task.

### 2.3 The NWP pipeline exists but is unused

`src/features/dataset.py` downloads `nwp_features.parquet` and discards it
("NOT used in training"). Correct as a finding; the fix in the superseded
plan does not work (§9.1).

### 2.4 Architectural details

- **No LayerNorm anywhere.** `_RelPosBipartiteConv` / `_RelPosSelfConv`
  docstrings state residual connections make training "stable without layer
  normalisation". GraphCast normalises every MLP. With `history_steps: 24`
  the METAR projection input is 1,152-dim unnormalised.
- **Absolute prediction, not residual.** Both reference papers predict a
  *delta* from current state. CropOS predicts absolute probability from
  scratch, so it has no skill floor.
- **`pos_weight` tuning history** (16.5 → 3.0 → 2.5 across commits `7821806`,
  `e976f52`) is calibration-chasing on a model with no signal. It is a
  symptom, not a cause.

### 2.5 Deviation from the cited papers, stated plainly

`src/models/gnn.py` cites **arXiv:2410.12938**, not Keisler or GraphCast; the
handover's "theoretical basis" is a retrofit. Against Keisler/GraphCast the
current model deviates on every structural axis: no mesh, no autoregression,
no vertical levels, no residual prediction, no normalisation. The handover's
claim that `local_mp_steps: 4` "corresponds to M=4 in the paper" does not
describe GraphCast, which uses 16 processor layers on a multi-mesh.

**Decision:** Keisler's *encode–process–decode with a multi-scale mesh* is
adopted (§3). Keisler's autoregressive state prediction is **not** adopted —
it requires pressure-level ERA5 that Open-Meteo cannot serve (§4.4), and a
GFS-correction framing serves the business case better.

---

## 3. Target architecture

### 3.1 Node types

| Type | Count | Source | Live at inference? |
|---|---|---|---|
| `grid` | ~1,980 | GFS fields, 22 vars incl. pressure levels, t−T…t | **Yes** |
| `station` | 16 (extensible) | METAR observations, 9 vars, t−T…t | **Yes** |
| `farm` | arbitrary | prediction targets, any lat/lon | n/a |

**ERA5 is removed from model inputs entirely.** It survives only as a
training label (§4.5). Rationale in §3.5.

`era5` is renamed `grid` and `metar` renamed `station` throughout, because
the node type is no longer tied to the data source. This is a breaking
rename; it is done now rather than later.

### 3.2 Encode–process–decode

```
encode:   grid nodes ──radius edges──▶ mesh nodes
process:  8 rounds of message passing on a MULTI-SCALE mesh
decode:   mesh ──▶ station nodes ──▶ farm nodes ──▶ residual head
```

**Multi-scale mesh.** Rather than build an icosahedron for a regional domain,
edges are constructed on one mesh node set at three ranges: nearest-neighbour
(~1°), medium (~4°), and long (~8°). This reproduces the property that makes
GraphCast's multi-mesh work — a message crosses the domain in ~4 hops instead
of ~40 — without the projection machinery of a global icosahedral grid.

**This is not optional.** Enlarging the domain (§3.3) without long-range
edges adds cost and no capability: 1,980 nodes connected only to their
neighbours has the same effective receptive field as 200. The two changes
must land together or neither is worth making.

**Processor depth:** 8 rounds (Keisler uses 6, GraphCast 16).

**LayerNorm in every MLP**, in both conv classes and the heads.

### 3.3 Domain — variable resolution

| Region | Box | Resolution | ≈ nodes |
|---|---|---|---|
| Thailand (local skill) | 5.5–20.5 °N, 97.5–105.7 °E | 0.5° | ~510 |
| Wider SEA (upstream) | 10 °S–30 °N, 85–125 °E | 1.0° | ~1,470 |
| **Total** | | | **~1,980** |

Fine where local skill is required, coarse where only upstream context is
needed — the same principle as the multi-mesh. The SEA box covers the Bay of
Bengal, Andaman Sea, Indochina, the South China Sea, and the equatorial band
where MJO signals originate: the actual upstream at 48 h lead.

Deduplicate the overlap: the 1.0° SEA grid excludes points falling inside the
Thailand box.

### 3.4 The GFS residual head

For horizon *h* ∈ {24, 48}, at sample time *t*:

```
gfs_h     = 10 surface GFS variables valid at t+h, forecast exactly h hours
            earlier  (_previous_day1 for h=24, _previous_day2 for h=48)
prior_h   = a · log1p(gfs_precip_mm @ t+h) + b        # a, b learned scalars
Δ_h       = MLP([farm_embedding, gfs_h, horizon_embedding])
logit_h   = prior_h + Δ_h
p_h       = sigmoid(logit_h)
```

Δ is initialised near zero (final layer weights zero-init). **The model
therefore begins training at GFS's own skill and learns only the
correction.** This is Keisler's predict-the-residual property and it supplies
a skill floor the current architecture lacks entirely: the model cannot
collapse far below the GFS baseline.

`a` and `b` are initialised from the Phase 0 logistic fit (§7) so the prior
starts calibrated rather than arbitrary.

### 3.5 Why GFS inputs rather than ERA5

Measured 2026-08-21 (§4.1): ERA5 via `archive-api` has ~1 day latency. Usable,
but a live forecast would run on a 24 h-stale atmospheric state, and the
train/serve distributions would differ.

GFS analysis-time fields are:

- **available live** at inference via `api.open-meteo.com/v1/forecast` with
  identical variable naming — zero train/serve mismatch;
- **richer vertically** — the 22-variable set includes 850/700/500/200 hPa
  wind, 850 hPa temperature and 500 hPa geopotential height, which the
  7-variable ERA5 surface set lacks;
- **the honest baseline** for the ablation (§5): arm A must represent what a
  farmer can actually obtain today, and ERA5 is not that.

Cost: the archive starts 2021-04, not 2016 (§4.1). Accepted.

### 3.6 Farm nodes decoupled from station nodes

`CropOSDataset` currently builds `farm_node_list` from `station_coords`, so
farm nodes are pinned to the 16 airports. Farm nodes must accept an arbitrary
list of `(lat, lon)` independent of the station set, so a farm sensor can be
sited anywhere without retraining. Default remains the 16 airport locations,
preserving current behaviour.

---

## 4. Data — measured availability

All figures below were verified by direct API probe on 2026-08-21, not taken
from documentation.

### 4.1 Archive start dates (**corrects a live bug**)

| Endpoint | Model | Verified start |
|---|---|---|
| `historical-forecast-api` | `gfs_seamless` | **2021-04** |
| `previous-runs-api` `_previous_dayN` | `gfs_seamless` | **2024-02** |
| `single-runs-api` | `gfs_seamless` | 2026-04 |
| `archive-api` ERA5 surface | — | 1940 → yesterday |
| `archive-api` ERA5 pressure levels | — | **all null — unavailable** |

> **BUG:** `configs/data.yaml` sets `nwp_min_start: "2016-01-01"` and
> `src/ingestion/nwp_baseline.py` documents "available from 2016-01-01
> onward". Both are wrong. Probes at 2016/2018/2020/2021-03 return 0/24
> non-null; 2021-04 returns 24/24.
>
> Consequence had the superseded plan been executed: five years of null rows
> per station, converted to zeros by `np.nan_to_num(arr, nan=0.0)` in
> `dataset.py`, training silently on fabricated data with no error raised.
>
> **Fix `nwp_min_start` to `2021-04-01` regardless of what else is built.**

### 4.2 What `historical-forecast-api` actually returns

Not lead-time-resolved forecasts. Per Open-Meteo: "a continuous hourly
timeseries built by stitching the **first hours** of each successive model
run." The row at valid time *V* is a ~0–6 h forecast from a run initialised
shortly before *V*.

Consequences:

- Used at times **≤ t**: legitimate. It is an analysis-quality field that is
  genuinely available at time *t* in production. This is how §3.1 uses it.
- Used at time **t+h**: **target leakage**. That value came from a model run
  initialised *after* t, which does not exist at inference. Validation BSS
  would rise sharply and production would fail.

This is a one-line distinction between a correct implementation and a
silently broken one. It must be stated in code comments at the point of use.

### 4.3 Lead-resolved GFS: what survives

`previous-runs-api` supplies values forecast exactly N×24 h before valid
time. Probing all 22 configured variables under `_previous_day2`:

- **10 usable:** `temperature_2m`, `dewpoint_2m`, `relativehumidity_2m`,
  `precipitation`, `windspeed_10m`, `winddirection_10m`, `surface_pressure`,
  `cape`, `lifted_index`, `shortwave_radiation`
- **4 null for GFS:** `cloudcover_low/mid/high`, `soil_moisture_0_to_7cm`
- **8 rejected outright:** all pressure-level variables. Not a null — a
  parser error: `Cannot initialize SurfacePressureAndHeightVariable… from
  invalid String value windspeed_850hPa_previous_day2`. The `_previous_dayN`
  suffix exists only for surface variables.

Lead offsets are whole days only, which is why horizons are **[24, 48]**:
`_previous_day1` and `_previous_day2` map onto them exactly. 12 h and 36 h
would require `single-runs-api`, whose archive begins 2026-04 — five months,
too short to train on, and ~150,000 requests to cover the period.

The pressure-level variables are therefore unavailable *at lead time* but
fully available *at analysis time*, which is where §3.1 places them.

### 4.4 Keisler-faithful autoregression is not buildable here

It requires ERA5 on pressure levels. `archive-api` returns all nulls for
`temperature_850hPa`, `geopotential_height_500hPa`, `windspeed_850hPa`,
`relativehumidity_700hPa`, `vertical_velocity_500hPa` at every date and under
every `models=` value tried. Obtaining them means a Copernicus CDS account
and a gridded download — out of scope, recorded here so the option is not
rediscovered later.

### 4.5 Labels

- **Training label:** ERA5 `precipitation` at the grid point nearest each
  farm node, thresholded at 1.0 mm — unchanged from current behaviour. Dense
  and gap-free.
- **Reporting label:** METAR `precip_mm` at the station, with a validity
  mask. Headline BSS is reported against real observations.

Labels are required only offline, so ERA5's latency is irrelevant to
production usability. **No production input depends on ERA5.**

`dataset.py` already warns when METAR `zero_frac > 0.60`. The masked METAR
evaluation must report its own coverage alongside every metric so a
high-coverage and a low-coverage station are never averaged silently.

### 4.6 Splits

| Split | Window | ≈ hours | Residual head |
|---|---|---|---|
| Pretrain | 2021-04 → 2024-01 | ~24k | inactive |
| Fine-tune | 2024-02 → 2025-06 | ~12k | active |
| Validation | 2025-07 → 2025-12 | ~4k | active |
| Test | 2026-01 → 2026-06 | ~4k | active |

Test is touched once, at the end.

---

## 5. The ablation — primary deliverable

Three arms, identical in every respect except the station branch.

| Arm | Station nodes | Predicts at | Measures |
|---|---|---|---|
| **A** | none | 16 airport locations | no local sensors at all |
| **B** | all 16 | same 16 locations | own sensor + neighbour network |
| **C** | 15, target masked | the masked station | neighbour network alone |

Derived quantities:

- **C − A** = value of a *network* of neighbouring stations
- **B − C** = value of a sensor **at your own site**
- **B − A** = total value

These price two distinct products. Report all three with confidence
intervals (bootstrap over test timestamps), per horizon, and per station.

**Validity constraints — these are requirements, not guidance:**

1. A and B share grid data, splits, seed, optimiser, schedule, and epoch
   budget. The *only* difference is the presence of the station branch.
2. Arm C is produced by evaluating the trained arm-B model with the target
   station's node masked — not by training 16 models.
3. Arm C is legitimate **only because** arm B trains with station DropNode,
   which makes "this station is absent" an in-distribution condition. Without
   it, C measures an out-of-distribution artefact.

### 5.1 `metar_dropout`, corrected and repurposed

`metar_dropout: 0.4` currently drops station-node embeddings after
projection. Two changes:

1. **It must apply to the station branch only, never to the GFS branch.**
   Under the superseded plan's design (§9.1), 40 % of samples would have had
   their GFS features zeroed — regularising away the signal being added.
2. Its purpose is now to make arm C valid, and that must be documented at the
   definition site. Its value becomes an experimental parameter: too high and
   arm B understates the value of stations; too low and arm C becomes
   out-of-distribution. Tune on validation, report the chosen value.

---

## 6. Corrections carried over from the superseded plan

Beyond the `nwp_min_start` bug (§4.1) and DropNode scoping (§5.1):

- **`history_steps` mismatch.** `configs/model.yaml` sets 24; the handover
  assumes 6 (`metar_in=288 = 48×6`). With 24, `train.py` computes
  `metar_in = 48 × 24 = 1152`. The smoke test as specified could not pass.
  Set explicitly in the new config and assert it in a test.
- **Static features must be projected once.** The 11 static columns
  (elevation, coast distance, terrain one-hots ×5, region one-hots ×4) are
  constant in time. Stacking them ×`history_steps` produces 264 duplicated
  dimensions, and the one-hot columns hit `FeatureScaler`'s 1e-8 std floor.
  Project static features once and concatenate after temporal stacking.

---

## 7. Phase 0 — measure the bar before building

Half a day, before any architecture work.

1. Fetch `precipitation_previous_day1` and `_previous_day2` for the 16
   stations, 2024-02 → 2026-06, from `previous-runs-api`.
2. Fit a 1-D logistic calibration from GFS `precip_mm` to rain probability on
   the fine-tune window.
3. Score raw and calibrated GFS at 24 h and 48 h against **both** label
   sources, as BSS vs climatology.

**This number is the bar.** Everything in §3 exists to beat it, and it seeds
`a` and `b` in §3.4.

**Gate:** if calibrated GFS BSS at 48 h is near zero or negative over
Thailand, the residual-head framing has little to correct and the design must
be revisited before the grid download is commissioned. Stop and report rather
than proceeding.

---

## 8. Ingestion

Gridded GFS is the largest build cost: ~1,980 points × ~46k hours × 22
variables ≈ **8 GB** as float32 (~10 GB on disk as parquet with index
columns).

- Extend the per-station checkpoint pattern already in
  `nwp_baseline.py::fetch_all_stations` to **grid tiles**; a partial download
  resumes rather than restarts.
- Multi-location bulk requests are supported (verified: comma-separated
  `latitude`/`longitude`, 3 locations returned in one call). Batch grid points
  per request.
- **Coverage verification is mandatory, not optional.** After every fetch,
  assert non-null fraction per variable per tile against an explicit
  threshold and **fail loudly**. The `nwp_min_start` bug (§4.1) would have
  been caught immediately by such a check, and `np.nan_to_num(…, nan=0.0)` in
  `dataset.py` is precisely what allowed it to pass silently. Audit every
  `nan_to_num` call and require an accompanying validity mask or an explicit
  justification.

---

## 9. Disposition of `2026-08-14-nwp-forecast-integration.md`

**Do not execute.** Retain for reference.

### 9.1 Why

The plan indexes NWP at `past_ts = ts - step * 1h` (plan lines 454–520) while
labels are at t+12…t+48. **The GFS forecast valid at the target time is never
read.** A GFS-correction model that never sees the forecast it corrects is
not a correction model. Expected effect: some gain at 12 h from better
analysis-time variables (CAPE, 850 hPa wind), approximately none at 48 h.

Combined with §4.1 (five years of nulls → zeros), §5.1 (DropNode destroying
40 % of the new signal), and §6 (`history_steps`, static stacking), the plan
would have produced a model that is larger, slower, trained partly on
fabricated data, and no more skilful.

### 9.2 What it got right, and is carried forward

- The diagnosis that NWP data is downloaded and discarded.
- `prepare_nwp_features()` and the 39-feature layout — reused at analysis
  time for grid and station nodes.
- The per-station checkpointing pattern — extended to grid tiles (§8).
- Backward-compatible optional-argument style for dataset construction.

---

## 10. Evaluation protocol

- **Primary:** BSS at 24 h and 48 h, against calibrated GFS as reference
  forecast — not only against climatology. "Better than the free public
  forecast" is the claim that matters.
- **Secondary:** reliability diagrams, POD/FAR/CSI at the operational
  threshold, and the agricultural decision categories already in
  `src/evaluation/agri_classifier.py`.
- **Ablation:** §5, with bootstrap confidence intervals, per horizon and per
  station.
- **Coverage:** METAR label coverage reported next to every METAR-scored
  metric.
- Test window is evaluated once.

---

## 11. Out of scope

Recorded so they are not silently reintroduced:

- Autoregressive state prediction (§4.4 — data unavailable).
- 12 h and 36 h horizons (§4.3 — no lead-resolved source deep enough).
- Copernicus CDS ingestion.
- `single-runs-api` for training (5-month archive; viable for inference later).
- Ensemble / probabilistic GFS (GEFS members).
- Re-tuning `pos_weight` before the residual head exists — §2.4.

---

## 12. Open questions for implementation

1. Exact mesh construction: mesh nodes coincident with grid nodes, or a
   separate coarser node set? Coincident is simpler; separate is closer to
   Keisler. **Decide during planning; default to coincident for v1.**
2. `metar_dropout` value — experimental parameter, tune on validation (§5.1).
3. Grid-point batch size per bulk API request — determine empirically against
   rate limits during Phase 0.
4. Whether arm A needs its own hyperparameter tuning to be a fair baseline,
   or must share arm B's settings for comparability. **Default: share
   settings**; document the choice, since it is the most likely challenge to
   the ablation's validity.
