# Enabling RRQPE on a pipeline + render-pool deployment

Notes for turning on the NOAA Enterprise Rain Rate (RRQPE) global observed
layer (`LIBREWXR_RRQPE_ENABLED=true`) on an existing deployment that runs it
disabled. Measurements were taken locally (Apple Silicon, macOS, Python 3.13,
NumPy 2.5, OpenCV 5.0) against live NOAA scans from 2026-10-01. They are
indicative: re-check on the target host before and after the switch.

## Fork changes this note depends on

- **Decode scaling fix.** `RRQPEGrid._decode_to_encoded` read the `RRQPE`
  variable with netCDF4 auto mask-and-scale on, which already returns mm/h,
  and then multiplied by `scale_factor` (0.1) a second time. Every rate was
  10x too low, which is -16 dBZ after the Marshall-Palmer conversion. On
  2026-10-01 04:20-04:40 UTC scans at the default downsample, the fix raises
  the maximum from 40.5 to 56.5 dBZ and the number of 0.04° cells above the
  default 10 dBZ noise floor from ~0.96 M to ~1.50 M (+57%). The previous
  test fixture made the inverse mistake on write, so the two cancelled.
- **Banded decode.** The packed int16 grid is now decoded in row bands
  aligned to the HDF5 chunk height (813 rows) and the downsample factor.
  Output is bit-identical to a whole-grid decode at factors 1, 2 and 4 on
  the live scans. Peak memory per 4.2 MB file falls from 2.34 GiB
  (NumPy-tracked; ~2.5 GiB RSS) to 0.33 GiB (~0.86 GiB RSS, which includes
  allocator retention on macOS), and decode time from 2.3 s to 1.6 s.
- **Point-nowcast `source`.** `/v2/radar/point-nowcast.json` now names the
  region that answered and what it observes (see below), so a satellite
  estimate is never presented as radar.

## What RRQPE stores

The region is 9000 x 3250 uint8 cells at the default
`LIBREWXR_RRQPE_DOWNSAMPLE=2` (0.04°), **27.9 MiB per frame**. That is larger
than OPERA (16 MiB) but smaller than USCOMP (63 MiB). All of the current
Doppler regions together are ~107 MiB per frame, so RRQPE adds ~26%.

For `FETCH_INTERVAL=300`, `MAX_FRAMES=12`, `NOWCAST_FRAMES=12` and
`RADAR_ANIMATION_SUBSTEPS=2`:

| Store | Files | Size | Notes |
|---|---|---|---|
| Scan store (`<cache>/rrqpe`) | 11-12 scans | ~310-335 MiB | Window `max_frames x interval + 2 x tolerance + 600 s`, ending 15 min before now |
| FrameStore (`<cache>/radar`) | 12 + 3 grace | ~420 MiB | One copy per 5-min frame (two consecutive frames share a scan) |
| Nowcast frames | 12 | 335 MiB | Rewritten every cycle |
| Animation frames | 11 past + 12 forecast | ~640 MiB | Past intermediates accumulate one per cycle; forecast ones are rewritten |
| State generations | 2 superseded sets | ~1.4 GiB | The pipeline keeps 3 published generations as hardlinks, so the previous two cycles' nowcast and forecast-animation files stay on disk |
| Coverage mask | 1 | ~90 MiB | RRQPE band mask in `<cache>/coverage_masks` |
| **Total unique files** | | **~3.2 GiB** | File-backed: disk plus page cache |

These are file pages. They count toward `docker stats` and the pipeline
container's `memory.current` (the pipeline writes them, so it is charged for
them), but the kernel reclaims them under pressure, so size from anon
(`memory.stat` → `anon`). A pipeline cgroup that already runs at its limit
reclaims more often. Pages that render workers read may then be evicted and
re-read from disk, so watch tile latency. `LIBREWXR_RRQPE_DOWNSAMPLE=4`
(0.08°) cuts every row of this table to about a quarter. Scan files are
4.2-4.5 MB each (~0.6 GB/day of S3 egress).

## Transient (anon) peaks in the pipeline

Each phase was measured on RRQPE alone in a fresh process:

| Phase | When | Wall | NumPy-tracked peak | RSS delta (macOS) |
|---|---|---|---|---|
| Decode one scan (after fix) | Each new scan, ~every 10 min; ~11 sequentially at a cold start | 1.6 s | 0.33 GiB | ~0.86 GiB |
| Decode one scan (before fix) | | 2.3 s | 2.34 GiB | ~2.5 GiB |
| Nowcast, 12 steps (distinct scans) | Every cycle | 6.8 s | 1.0 GiB | ~1.9 GiB |
| Nowcast, 12 steps (same scan twice) | Every other cycle at 5 min | 4.4 s | 1.0 GiB | ~1.5 GiB |
| Animation intermediates (substeps 2) | Every cycle | 1.1 s | 0.64 GiB | ~0.3 GiB |

The nowcast cost comes from the wrap-padded warp that global regions use:
`_extrapolate_forward(..., wrap=True)` still builds two full float32
coordinate grids, plus an int64 `mgrid` transient, for every step. Regional
radars use OpenCV relative maps and avoid that. Nowcast steps run on a
4-thread pool, so the RRQPE task overlaps the USCOMP/OPERA tasks of the same
step. Budget **~1-2 GiB extra transient anon** in the pipeline during nowcast
generation and **<1 GiB** while a scan decodes. Steady-state anon rises only
by tens of MiB: flows are stored at ≤1000 px and everything else is memmaps.

An isolated end-to-end pilot measured the whole pipeline: +~1.0 GiB at the
lifetime peak and +~90 MiB median steady state against the same
configuration with RRQPE off. See [Pilot results](#pilot-results).

Render workers map the same memmap files read-only. RRQPE tiles render
in-process from the 27.9 MiB grid. On the pilot a cold 256 px tile took p50
9 ms / p95 23 ms, and point-nowcast answered in 1-2 ms.

## Does RRQPE take part in the nowcast?

Yes. The wall-clock gate in `RRQPEGrid.match_timestamp` only stops RRQPE from
serving *scans* for future timestamps. Once observed frames are in the
FrameStore, the optical-flow nowcast extrapolates RRQPE like any other radar
region (`Nowcast generation: 12 frames x N regions` counts it). The animation
interpolates it too. With `NOWCAST_BLEND_MODE=radar` every forecast step up
to 60 min has blend weight 1.0. Where RRQPE is the only observed layer, the
nowcast tiles and point-nowcast forecast frames therefore come purely from
extrapolating it.

Two caveats matter to clients:

1. **Constant 30-minute lag.** A frame labelled `T` carries the scan from
   `T - 30 min`; at a 5-min fetch interval, odd slots carry `T - 35 min`.
   The "current" RRQPE frame is a 30-35 minute old estimate, and a forecast
   frame labelled `T + k` is that scan extrapolated by `k`. Physically it
   describes roughly `T + k - 30 min`.
2. **5-minute frames over 10-minute scans.** The nowcast computes flow
   between the two newest frames and treats it as motion per
   `FETCH_INTERVAL`. At 300 s, every other pair is the same scan twice:
   flow is zero, so the forecast is persistence. The pairs in between hold
   scans 10 min apart, which doubles the apparent speed. On the
   2026-10-01 04:30/04:40 UTC scans, wet-area motion had a median of
   58 km/h and a p90 of 228 km/h, and 15% of vectors hit the 200 km/h
   clamp. Pairing the same scans at their true 10-min spacing gives
   29 km/h median, 114 km/h p90 and 0.6% clamped. The RRQPE forecast
   therefore alternates between frozen and double-speed every cycle.

Recommendation: treat RRQPE forecast frames as low-confidence guidance, and
do not derive "rain starts/ends in N minutes" from them, until the nowcast
pairs RRQPE frames one native scan apart and scales the flow to the fetch
interval. The animation's newest-pair interpolation must keep using the pair
flow. At the upstream default `FETCH_INTERVAL=600` the cadence problem does
not occur; the 30-minute lag still applies.

## Point-nowcast contract

`/v2/radar/point-nowcast.json` gains an optional top-level `source` object.
Existing fields are unchanged and clients that ignore it keep working:

```json
"source": {
  "region": "RRQPE",
  "type": "satellite",
  "grid_spacing_km": 4.45,
  "nominal_lag_seconds": 1800,
  "attribution": "Precipitation data from NOAA Enterprise Rain Rate (RRQPE)"
}
```

- `type` is `radar` for every Doppler composite and `satellite` for RRQPE.
- `grid_spacing_km` is the coarser axis of the native cell at the requested
  latitude (RRQPE rows are 0.04° ≈ 4.45 km).
- `nominal_lag_seconds` is the fixed design offset between data time and
  frame time. It is `null` for regions that do not declare one; that means
  "not characterised", not "zero".
- `attribution` is the text the data provider requests, or `null`.
- `source` is `null` unless at least one returned frame is `in_range`. The
  region resolver falls back to a region's bbox when it has no coverage
  mask, so with `ENABLED_REGIONS=ALL` and RRQPE disabled, points outside
  Doppler coverage still resolve to `RRQPE`, with every frame
  `out_of_range`. Such responses do not claim a satellite source.

One region answers each request: the finest-resolution region whose
coverage contains the point *and* that has data in the observed window.
This matches the renderer, which falls through to the next coarser region
when a finer one has no array. A Doppler region with a disabled or failing
provider no longer shadows RRQPE. For example, Manila answers from RRQPE
while `PHCOMP` has no frames, then switches back to `PHCOMP` (`radar`) once
it does. If no covering region has data, the finest one is reported with
`out_of_range` frames. The MCP `get_precip_nowcast` tool picks its region
the same way, per frame, and reports `source: "satellite"` for RRQPE.

## Enabling and rolling back

The only configuration change is one line in `.env`, which the pipeline and
every renderer read:

```diff
-LIBREWXR_RRQPE_ENABLED=false
+LIBREWXR_RRQPE_ENABLED=true
```

Renderers need the flag as well as the pipeline: it adds the RRQPE coverage
polygon to their masks and RRQPE to their enabled-region list.

Procedure for the `docker-compose.yml + docker-compose.pool.yml (+ override)`
shape. Restart only the pipeline and renderers; leave the router, volumes
and any auto-update setting as they are.

1. Record the baseline: `free -m`, swap, `docker stats --no-stream`, each
   container's cgroup `memory.stat` (`anon`, `file`), `memory.peak` and
   `memory.events`, `/health` → `frames.per_region`, and the current image
   IDs. The `/health` `cluster.memory` block covers a render container only,
   not the pipeline. Tag those images for rollback, for example
   `docker tag librewxr-pipeline:latest librewxr-pipeline:before-rrqpe`, and
   the same for `librewxr-renderer` and `librewxr-renderer-secondary`.
2. Check out the fork revision that contains the fixes above. Run
   `pytest tests/test_rrqpe_grid.py tests/test_point_nowcast.py
   tests/test_point_nowcast_source.py`, then build
   the `pipeline`, `renderer` and `renderer-secondary` images.
3. Apply the `.env` change.
4. `up -d --no-deps pipeline`. Wait for `RRQPE: N new scan(s)` and the first
   `Nowcast generation` log line (about 30-60 s; the cold backfill decodes
   ~11 scans).
5. Roll `renderer-secondary`, check readiness through the router, then roll
   `renderer`.
6. Verify: `/health` → `frames.per_region.RRQPE` grows to `MAX_FRAMES`;
   point-nowcast for points outside Doppler coverage returns
   `source.type == "satellite"` with `in_range` frames; points inside
   Doppler coverage keep their radar region.

Rollback has two levels:

- **Configuration only** (the code fixes stay): set
  `LIBREWXR_RRQPE_ENABLED=false` and repeat steps 4-5. RRQPE disappears
  from new frames. Old frames age out of the 12-frame window within an hour,
  and point-nowcast returns `source: null` for those points again.
- **Code**: retag the `before-rrqpe` images as `latest` and
  `up -d --no-deps --force-recreate` the same three services.

Either way, `<cache>/rrqpe` can be deleted afterwards to reclaim ~330 MiB of
disk.

## What to watch for 24 hours

- **Memory.** Pipeline anon (cgroup `memory.stat` → `anon`) at nowcast
  time, and the growth rate of the pipeline's `memory.events` `max`
  counter (reclaim at the limit). Look for OOM kills
  (`docker inspect -f '{{.State.OOMKilled}}'`, `memory.events` `oom_kill`,
  `dmesg`), host swap growth, and renderer anon against the pool limit.
- **Latency.** Tile p50/p95 and point-nowcast latency through the router or
  cache, compared with the baseline.
- **Freshness.** `/health` → `frames.per_region.RRQPE` stays at
  `MAX_FRAMES`. The pipeline log shows `RRQPE: 1 new scan(s)` about every
  10 minutes; listing or download warnings are retried next cycle.
- **Spot checks.** Points without Doppler coverage (New Delhi, Lagos,
  São Paulo, Nairobi, Jakarta) report the satellite source. Doppler cities
  are unchanged.

## Attribution

NOAA requests: "Precipitation data from NOAA Enterprise Rain Rate (RRQPE)".
No NOAA endorsement is implied, and the data must not be presented as
unaltered NOAA data. Show it wherever RRQPE-derived tiles or point estimates
are displayed.

## Pilot results

Two isolated local deployments ran side by side for 10 fetch cycles
(2026-10-01 05:47-06:30 UTC). Each was `python -m librewxr.main` with an
auto-spawned pipeline, its own cache directory and a localhost port, using
`ENABLED_REGIONS=SVCOMP`, `FETCH_INTERVAL=300`, `MAX_FRAMES=12`,
`NOWCAST_FRAMES=12`, `NOWCAST_BLEND_MODE=radar`,
`RADAR_ANIMATION_SUBSTEPS=2`, and ECMWF, regional NWP, satellite and alerts
off. The only difference was `LIBREWXR_RRQPE_ENABLED`. Memory is macOS
`phys_footprint` (dirty/anon-like; excludes clean file pages), sampled every
10 s.

| | RRQPE on | RRQPE off |
|---|---|---|
| Pipeline footprint, median after warm-up | 285 MiB | 197 MiB |
| Pipeline footprint, max sample (during nowcast) | 1209 MiB | 198 MiB |
| Pipeline footprint, lifetime peak (incl. cold backfill) | 1268 MiB | 282 MiB |
| Pipeline RSS max (incl. touched file pages) | 2871 MiB | 303 MiB |
| Render worker footprint | 145 MiB | 143 MiB |
| Nowcast generation per cycle | 4.4-7.4 s | 0.1 s |
| Unique files in the cache dir | 3270 MiB | 55 MiB |
| Cold backfill | 11 scans in ~25 s | - |
| Cold 256 px tile, p50 / p95 (40 tiles at z3-z9) | 9 / 23 ms | 0.4 ms (empty-tile path) |
| Point-nowcast | 1-2 ms | <1 ms |

`/health` reported `frames.per_region.RRQPE = 12` throughout. The pipeline
logged one new scan every other 5-min cycle, with no warnings. New Delhi,
Lagos, São Paulo, Nairobi and Jakarta all returned
`source.type == "satellite"` with `in_range` frames. In a production
`ENABLED_REGIONS=ALL` set (resolver check), Sydney and Cape Town also
resolve to RRQPE; Berlin, Vilnius and Murmansk resolve to OPERA. Manila
resolves to PHCOMP by coverage and falls through to RRQPE while PHCOMP has
no frames.

On a real deployment, RRQPE tiles replace IFS fill rather than the
empty-tile path, so compare tile latency against the deployment's own
baseline.
