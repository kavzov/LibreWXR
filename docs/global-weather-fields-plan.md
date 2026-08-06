# Global Weather Fields — Architecture and Implementation Plan

> **Status:** design only; no runtime code has been changed. Research was
> performed against the repository at `df884c3` and the live Open-Meteo IFS
> dataset on 2026-08-06.

## Goal and invariants

Add global raster layers for near-surface temperature, dew point, mean sea-level
pressure, and 10 m wind without changing the existing radar, satellite, alerts,
nowcast, MCP, or Rain Viewer-compatible behaviour.

The implementation must preserve these invariants:

- `/public/weather-maps.json`, `/v2/radar/...`, `/v2/satellite/...`,
  `/v2/coverage/...`, and `/v2/alerts` keep their current schemas, URLs,
  cache semantics, and rendering behaviour.
- The new layers cover the complete slippy-map world, including oceans and
  areas outside every radar or regional NWP domain.
- A model generation becomes visible only after every configured field and
  valid time in that generation has been decoded, validated, flushed, and made
  readable.
- The previous complete generation remains active while a new one downloads
  and whenever the new one fails validation.
- A missed or incomplete upstream run never replaces usable data with empty
  tiles.
- Adding a later scalar or vector field is a registry/specification change,
  not another store/API/renderer refactor.
- The raster pipeline reads bulk model files. It must not build global rasters
  by fanning out requests to Open-Meteo's point forecast API.

## Research result: the current Open-Meteo dataset is sufficient

### Live `latest.json`

LibreWXR currently reads
[`data_spatial/ecmwf_ifs/latest.json`](https://openmeteo.s3.amazonaws.com/data_spatial/ecmwf_ifs/latest.json)
from the anonymous `openmeteo` bucket in `us-west-2`. At the time of this
investigation it reported:

- `completed: true`
- `reference_time: 2026-08-06T06:00:00Z`
- `last_modified_time: 2026-08-06T11:56:34Z`
- O1280 reduced Gaussian grid covering `[-90, -180, 90, 180]`
- hourly valid times through T+90, then three-hourly valid times in the part of
  the horizon inspected
- the following 35 values in `variables`:

```text
boundary_layer_height
cape
cloud_cover
cloud_cover_high
cloud_cover_low
cloud_cover_mid
convective_inhibition
dew_point_2m
direct_radiation
precipitation
pressure_msl
shortwave_radiation
showers
snowfall_water_equivalent
soil_moisture_0_to_7cm
soil_moisture_100_to_255cm
soil_moisture_28_to_100cm
soil_moisture_7_to_28cm
soil_temperature_0_to_7cm
soil_temperature_100_to_255cm
soil_temperature_28_to_100cm
soil_temperature_7_to_28cm
surface_temperature
temperature_2m
temperature_2m_max
temperature_2m_min
total_column_integrated_water_vapour
visibility
wind_gusts_10m
wind_u_component_100m
wind_u_component_10m
wind_u_component_200m
wind_v_component_100m
wind_v_component_10m
wind_v_component_200m
```

The exact stored wind names are therefore
`wind_u_component_10m` and `wind_v_component_10m`, not `wind_u_10m` and
`wind_v_10m`. Open-Meteo exposes derived API names such as `wind_speed_10m`
and `wind_direction_10m`, but stores the native vector components. Its
[ECMWF documentation](https://open-meteo.com/en/docs/ecmwf-api) explicitly
describes wind speed/direction as derived from U/V and says ECMWF supplies 2 m
dew point rather than relative humidity.

### Physical contents of the `.om` files

The investigation opened the same bulk object that the current precipitation
loader would use for T+1:

```text
data_spatial/ecmwf_ifs/2026/08/06/0600Z/2026-08-06T0700.om
```

`OmFileReader` reported all of these as sibling float32 arrays with shape
`(1, 6599680)` in that one file:

| Required LibreWXR field | Actual Open-Meteo child | Present with precipitation | Values returned by `omfiles` | Public/API unit |
|---|---|---:|---|---|
| 2 m temperature | `temperature_2m` | yes | °C | °C |
| 2 m dew point | `dew_point_2m` | yes | °C | °C |
| mean sea-level pressure | `pressure_msl` | yes | Pa | hPa after `/ 100` |
| 10 m eastward wind | `wind_u_component_10m` | yes | m/s | m/s internally |
| 10 m northward wind | `wind_v_component_10m` | yes | m/s | m/s internally |
| precipitation | `precipitation` | yes | mm for the preceding interval | mm |

Small range reads confirmed plausible decoded values: temperatures and dew
points were in degrees Celsius, pressure was around `101160` Pa, and wind
components were in m/s. The direct ECMWF names for the same parameters are
`2t`, `2d`, `msl`, `10u`, and `10v`; they are also present in the
[ECMWF Open Data catalogue](https://www.ecmwf.int/en/forecasts/datasets/open-data).

The T+0 analysis file was also inspected. It contains all five instantaneous
weather arrays but does **not** contain `precipitation` or
`snowfall_water_equivalent`. This agrees with the
[Open-Meteo Open Data layout documentation](https://github.com/open-meteo/open-data/blob/main/README.md):
backward sums/averages omit their first timestep, while `data_spatial` places
the variables available for a valid time in one `.om` object.

### Source decision

Use the existing Open-Meteo `data_spatial/ecmwf_ifs` dataset and the same
per-valid-time `.om` files. Do not use the point forecast API. Do not introduce
a second Open-Meteo dataset path, and do not switch the first implementation to
direct ECMWF GRIB.

Direct ECMWF Open Data GRIB remains a viable fallback because it publishes all
five native parameters, but it would add GRIB index/range selection, unit
conversion from Kelvin, a different accumulation/cadence contract, and another
download implementation without providing fields that the current `.om`
objects lack. It should be reconsidered only if Open-Meteo removes required
variables, changes its redistribution terms, or proves unable to meet latency
or availability requirements.

## Current architecture

### Ingest and time selection

`src/librewxr/sources/world/ifs/grid.py:ECMWFGrid` is both an ingest object and
the global precipitation implementation of `NWPSource`.

1. `fetch()` moves synchronous work to a thread and retains already-loaded data
   when `_fetch_sync()` fails before publication.
2. `_fetch_sync()` reads `latest.json`, requires `completed`, requires the
   `precipitation` variable, and derives the run prefix from `reference_time`.
3. It deliberately passes `valid_times[1:]` to `_select_valid_times()`, because
   T+0 has no accumulated precipitation.
4. `_select_valid_times()` chooses a window around the radar history and
   nowcast horizon. The count is controlled by
   `settings.get_ecmwf_max_timesteps()`.
5. `_fetch_one_timestep()` opens one `.om`, reads `precipitation` and optionally
   `snowfall_water_equivalent`, and uses `earthkit.regrid.interpolate` to map
   O1280 onto the regular global 0.1° grid (`1801 x 3600`).
6. Accumulated precipitation is interpreted as the preceding-hour rate,
   classified as rain/snow, converted through the configured Z-R relationship,
   and encoded as radar-compatible uint8 dBZ.
7. Hourly precipitation and snow masks may be optical-flow-interpolated onto
   the ten-minute radar cadence.
8. Arrays are written to memmaps and `_timesteps`, `_sorted_timestamps`, and
   `_reference_time` are replaced.

`RadarFetcher._fetch_auxiliary_grids()` calls every NWP contribution under the
shared `nwp_fetch_concurrency` semaphore. IFS is registered by
`sources/world/ifs/__init__.py:nwp_provider()` with priority 1000 and snapshot
slug `ecmwf_grid`.

### Runtime model and precipitation render

`data/nwp_source.py:NWPSource` is intentionally precipitation-specific:

- `sample()` returns radar-compatible uint8 dBZ;
- `get_snow_mask()` returns a boolean phase classification;
- domain/feather methods drive regional-to-global compositing;
- `NWPChain` blends model precipitation and performs first-source snow
  dispatch.

`tiles/renderer.py:compute_tile_geometry()` composites radar, uses
`NWPChain.sample()` outside radar coverage or in a nowcast blend, applies the
noise floor, and optionally samples the snow mask. It returns a cacheable
`TileGeometry`. `present_tile()` performs colour lookup, blur, overlays, and
PNG/WebP encoding. `png_palette.py:encode_png()` already provides deterministic,
lossless palette PNGs when the rendered RGBA result has at most 256 colours.

This contract must not be generalized into arbitrary floats. Temperature,
pressure, and wind have different units, missing-value rules, temporal
semantics, and presentation logic. Changing `NWPSource.sample()` would put the
existing radar/nowcast path at unnecessary risk.

### Storage and shared state

IFS precipitation uses one memmap pair per stored timestamp. `ECMWFGrid` is
serializable through `__getstate__()` and reopens those files in
`__setstate__()`.

In single mode, `main.lifespan()` creates the stores, the fetcher, renderer
executors, tile cache, and API globals in one process. If `cache_dir` is set it
also dumps `state.json` for the standalone MCP process.

In multi mode:

- `data_pipeline.run_pipeline()` owns downloads and decoding;
- `on_cycle_complete()` builds the global precipitation mask and calls
  `master_state.dump_state()`;
- `dump_state()` writes `.state.json.tmp` and atomically renames it to
  `state.json`;
- render-only workers poll the mtime, load the JSON off the event loop, and
  call each store's `__setstate__()` in place;
- `_compute_cache_invalidation()` fully clears the radar tile cache if an NWP
  reference time or IFS timestep set changes.

The JSON file is atomic, but the current IFS on-disk generation is not a
complete transactional unit: `_cleanup_memmap_files()` unlinks old names before
all replacement memmaps are written. Open mappings remain usable on Unix, but
a worker starting/reloading in that interval, a crash, or a non-Unix filesystem
can observe missing files. The new field store must not copy this pattern.

### API and cache

`api/routes.py` holds lifespan-wired module globals. Existing metadata is a
strict Rain Viewer-shaped `WeatherMapsResponse`; it must not receive the new
catalogue. Radar tiles use two entries in `TileCache`:

- geometry key `(timestamp, z, x, y, size, smooth, snow)`;
- encoded/present key `(timestamp, z, x, y, size, smooth, snow, colour, ext,
  webp_quality)`.

`TileCache` is a thread-safe byte-capped LRU. Its current capacity and eviction
behaviour directly affect radar latency, so global field tiles should use a
separate instance and memory budget rather than compete with radar entries.

## Proposed architecture

### Separate weather-field subsystem, shared IFS reader

Introduce a generic weather-field subsystem alongside, not inside, the
precipitation `NWPChain`:

```text
Open-Meteo latest.json + per-valid-time .om
                    |
          IFSOpenMeteoRunReader
             /              \
 existing ECMWFGrid       GlobalWeatherFieldStore
 (dBZ + snow adapter)     (typed scalar/vector fields)
             |              |
       existing NWPChain    new weather-field API/renderer
```

The first implementation should extract URL construction, manifest parsing,
`OmFileReader` opening, O1280 regridding, and source-name validation from
`grid.py` into a small `IFSOpenMeteoRunReader`. `ECMWFGrid` continues to expose
the exact same precipitation methods and state shape. The weather-field loader
uses the shared reader but has its own time selection, staging, store, and
publication policy.

This avoids duplicate protocol logic while allowing precipitation and the new
fields to be rolled out and tested independently. In particular, the new
loader can use T+0 while the precipitation path continues to skip it.

### Universal field model

Define immutable specifications rather than field-specific classes:

```python
@dataclass(frozen=True)
class WeatherFieldSpec:
    id: str                         # stable public ID
    kind: Literal["scalar", "vector"]
    source_components: tuple[str, ...]
    unit: str                       # canonical stored/API unit
    storage_scale: float
    storage_offset: float
    temporal: Literal["linear", "vector_linear", "nearest"]
    palette: str
    display_range: tuple[float, float] | None
```

Initial registry:

| Public field ID | Kind | Source component(s) | Canonical unit | Temporal rule |
|---|---|---|---|---|
| `temperature_2m` | scalar | `temperature_2m` | °C | linear |
| `dew_point_2m` | scalar | `dew_point_2m` | °C | linear |
| `pressure_msl` | scalar | `pressure_msl` divided by 100 | hPa | linear |
| `wind_10m` | vector | `wind_u_component_10m`, `wind_v_component_10m` | m/s | component-wise linear |

Internally, a `WeatherFieldGeneration` contains `generation_id`,
`reference_time`, sorted native valid times, a spec/version fingerprint, and
memmap descriptors per `(field, component, valid_time)`. A
`GlobalWeatherFieldStore` owns exactly one active immutable generation and
optionally references the previous retained generation.

The store API should be independent of Open-Meteo:

```python
available_fields() -> list[WeatherFieldSpec]
available_times(field_id) -> list[int]
sample_scalar(field_id, lat, lon, timestamp, bilinear=True) -> ndarray
sample_vector(field_id, lat, lon, timestamp, bilinear=True) -> tuple[ndarray, ndarray]
has_generation() -> bool
generation_metadata() -> dict
```

Adding cloud cover, visibility, gusts, or another level later becomes a new
`WeatherFieldSpec` plus a palette. A derived field may supply a transform
function, but the store and public API remain unchanged.

### Storage, precision, and units

Keep the existing global regular grid:

- WGS84 longitude `[-180, 180)` with dateline wrap;
- latitude `[90, -90]` in 0.1° steps;
- shape `1801 x 3600`;
- O1280 source values regridded by the same tested earthkit path as
  precipitation.

Do not keep all fields as float32. One global float32 array is about 25.9 MB;
five component arrays per valid time would be about 129.7 MB before any
history. Use scale/offset int16 memmaps with `-32768` reserved for missing:

| Component | Canonical value | Proposed quantization | Approx. max quantization error |
|---|---|---|---|
| temperature | °C | 0.05 °C/count | 0.025 °C |
| dew point | °C | 0.05 °C/count | 0.025 °C |
| pressure MSL | hPa | 0.1 hPa/count with offset | 0.05 hPa |
| wind U/V | m/s | 0.05 m/s/count | 0.025 m/s |

This is roughly 13 MB per component per native valid time, about 65 MB for the
initial four public fields. Values are decoded to float32 only for requested
tile-sized samples. The generation manifest records dtype, scale, offset,
shape, source name, canonical unit, checksum/size, and missing sentinel so the
disk format is self-describing.

Use only native model times on disk. Unlike precipitation animation, do not
materialize ten-minute synthetic global arrays. Interpolate the two bracketing
native fields after spatial sampling; this reduces disk/RAM use by roughly the
native-to-display cadence ratio.

### Time window and interpolation

The weather-field window is independent of radar history:

- include T+0 for instantaneous fields;
- include a configurable forecast horizon, proposed default 48 hours;
- optionally keep two hours before wall-clock now when those times are still in
  the active run;
- preserve the upstream native cadence in the catalogue.

For a requested timestamp:

1. Find exact or bracketing native valid times with binary search.
2. Spatially sample each endpoint. Bilinear interpolation is the default;
   longitude wraps at the dateline and latitude clamps at the poles.
3. Linearly interpolate temperature, dew point, and pressure in physical
   units.
4. Interpolate wind U and V independently, then derive speed
   `sqrt(u**2 + v**2)` and meteorological direction. Never interpolate wind
   direction angles directly across 0°/360°.
5. If exactly one endpoint is missing at a pixel, use the valid endpoint. If
   both are missing, preserve missing/transparent. A globally complete run with
   an unexpectedly high missing fraction fails validation before publication.
6. Outside the stored valid-time interval, return 404 for that immutable run;
   do not silently clamp to an unrelated endpoint.

Optical flow is inappropriate for these continuous state variables. It remains
specific to precipitation echoes.

### Atomic generation publication

Use generation directories instead of timestamp filenames reused in place:

```text
<cache>/weather_fields/ecmwf_ifs/
  generations/
    20260806T0000Z-<spec-hash>/...
    20260806T0600Z-<spec-hash>/...
  staging/<uuid>/...
```

Publication sequence:

1. Fetch a completed `latest.json`. If its reference time is already active,
   fetch only if the configured time window/spec fingerprint changed.
2. Create a unique staging directory; never touch active generation files.
3. For every required valid time, open the `.om` once, read all configured
   source components, regrid, convert units, quantize, and write memmaps.
4. Validate component presence, exact shapes, finite/range limits, valid-time
   consistency, global coverage, and file sizes. Any missing required
   component rejects the entire candidate.
5. Flush memmaps, write the generation manifest last, and fsync files and the
   staging directory where supported.
6. Atomically rename the staging directory into `generations/`.
7. Build a complete in-memory `WeatherFieldGeneration`, then swap the store's
   single active-generation reference under a short lock. In-flight renders
   retain the old object and mappings.
8. Only after that swap may `dump_state()` publish a `state.json` referencing
   the new immutable directory.
9. Retain at least the active and immediately previous complete generations.
   Delete older generations only after a grace interval longer than worker
   polling plus the maximum render duration. Startup cleanup may remove stale
   staging directories, never a referenced generation.

For multi mode, `GlobalWeatherFieldStore.__getstate__()` serializes only the
generation manifest reference and public metadata. `__setstate__()` opens every
new memmap into a temporary generation object, validates that all files exist,
and swaps one reference only after the full reopen succeeds. If reopen fails,
the worker logs the error and retains its previous active generation.

`master_state.apply_state()` already catches a failed store refresh and leaves
the object alive. The weather store must also be treated like `frame_store`,
`precip_mask`, and `nowcast_store` during boot compatibility: a legacy snapshot
without it must not make later resurrection impossible. Add a generic
store-resurrection helper instead of another one-off function if practical.

### Failure and stale-data policy

- `completed=false`, unavailable `latest.json`, a transient range-read error,
  a corrupt component, or an incomplete candidate leaves the active generation
  untouched.
- Retry failed downloads according to `download_retries`; retries write only
  into staging.
- Persist the previous complete generation across restarts. Cold start should
  load it from its manifest before attempting a network refresh.
- There is no automatic age at which good data is replaced with no data.
  Catalogue and `/health` report `stale`, `age_seconds`, last attempt, and last
  error so operators can alert without breaking maps.
- If no complete generation has ever existed, the new catalogue/tile endpoint
  returns 503. It must not return a successful transparent tile, which would
  falsely claim that the weather field is empty.
- A field absent from a new run is a failed generation, not a partially
  published run. Optional future fields can be disabled in configuration, but
  all enabled fields form one atomic unit.

## API design

Keep the Rain Viewer document unchanged and add a separate LibreWXR extension.

### Catalogue

```http
GET /v2/weather-fields.json
```

Proposed response:

```json
{
  "version": "1.0",
  "generated": 1786017600,
  "source": "ECMWF IFS via Open-Meteo",
  "generation": "20260806T0600Z-a1b2c3d4",
  "reference_time": 1785996000,
  "stale": false,
  "fields": [
    {
      "id": "temperature_2m",
      "kind": "scalar",
      "unit": "°C",
      "times": [1785996000, 1785999600],
      "path": "/v2/weather/20260806T0600Z-a1b2c3d4/temperature_2m"
    },
    {
      "id": "wind_10m",
      "kind": "vector",
      "unit": "m/s",
      "components": ["u", "v"],
      "times": [1785996000, 1785999600],
      "path": "/v2/weather/20260806T0600Z-a1b2c3d4/wind_10m"
    }
  ]
}
```

### Tiles

```http
GET /v2/weather/{generation}/{field}/{timestamp}/{size}/{z}/{x}/{y}.{ext}
```

Query parameters:

- `smooth=0|1`, default `1`;
- `style=default` initially, reserved for stable named palettes;
- `vectors=0|1` for the `wind_10m` direction overlay, default `1`.

The generation in the path makes tile URLs immutable. The catalogue switches
all clients to a new URL only after atomic publication, and CDN entries for the
same timestamp cannot accidentally survive a run change. Return 404 for an
unknown field/time/generation, 400 for invalid tile coordinates, and 503 only
when the subsystem has no complete generation.

Do not expose the Open-Meteo component names as the primary vector API. The
catalogue records provenance, while the public `wind_10m` field presents speed
and direction as one coherent layer. A future raw-data format can expose U/V
explicitly without changing the visual tile contract.

Add Pydantic models such as `WeatherFieldsResponse`, `WeatherFieldMetadata`,
and `WeatherFieldSourceMetadata` in `api/models.py`. Do not add a
`weather_fields` member to `WeatherMapsResponse`.

### Rendering

Add `tiles/weather_field_renderer.py` rather than extending precipitation's
`TileGeometry`:

- `compute_weather_field_geometry()` builds tile lat/lon arrays, samples the
  store in physical units, interpolates time, and returns a
  `WeatherFieldGeometry` containing scalar values or U/V components plus a
  missing mask;
- `present_weather_field_tile()` applies a named LUT/range, optional isolines or
  wind vectors, and calls the existing `_encode_image`/`encode_png` path;
- temperature and dew point use fixed meteorological ranges/palettes;
- pressure uses a fixed synoptic range in hPa, with optional contours deferred
  unless the first UI requires them;
- wind colour represents speed while arrows represent direction;
- legends and palette metadata live in a registry so frontend clients can
  reproduce them.

Keep the physical-value geometry independent of output format and palette, as
the current precipitation compute/present split does. Continuous smoothing may
produce more than 256 RGBA colours; `encode_png()` will automatically select
RGBA in that case and palette PNG where it remains lossless. No change to
`png_palette.py` is expected.

## Cache strategy

Instantiate a separate byte-capped `TileCache` for weather fields so enabling
the feature cannot evict radar geometries or encoded Rain Viewer tiles.

Recommended keys:

```text
geometry:
(generation, field, timestamp, z, x, y, size, smooth)

present:
(generation, field, timestamp, z, x, y, size, smooth,
 style, vectors, ext, webp_quality)
```

Because generation is part of every key, cached tiles never mix model runs.
On a state refresh, clear only the weather cache when its generation changes;
do not trigger the existing full radar cache clear merely because the new
weather store changed. Let old-generation entries age out by LRU, or clear the
weather cache eagerly after the store swap.

Add a separate `LIBREWXR_WEATHER_TILE_CACHE_MB` budget and include it in memory
pressure handling and `/health`. The first rollout should be opt-in with
`LIBREWXR_GLOBAL_WEATHER_FIELDS_ENABLED=false`: the additional disk/RAM and
network cost is material, and opt-in guarantees unchanged resource behaviour
for existing installations. Flipping the default can be a later measured
decision.

Set immutable generation tile responses to a long public max-age (for example
86400 seconds) with ETags. Set the catalogue to a short max-age aligned with
the fetch interval. A retained old generation may continue serving its URLs
during the grace period; after cleanup it may return 404 while already-cached
CDN copies remain valid.

## Single- and multi-mode integration

### Single mode

- Construct `GlobalWeatherFieldStore` and its loader in `main.lifespan()` when
  enabled.
- Pass the contribution to `RadarFetcher` so it participates in the existing
  auxiliary fetch cycle and shutdown path, but keep its publication independent
  of precipitation success.
- Bind the store and a dedicated weather tile cache to `api.routes`.
- If `cache_dir` exists, include the store in the optional single-mode
  `state.json`; otherwise use a versioned temporary directory with the same
  atomic semantics.

### Multi mode

- The pipeline alone downloads, regrids, validates, and publishes generations.
- Include `weather_fields` in the pipeline `stores` dictionary and the atomic
  state snapshot.
- Render-only workers construct an empty store, apply/reopen its complete
  generation, bind it to routes, and poll for later generations.
- A worker unable to reopen the new generation continues serving its previous
  one and reports degraded health; other workers are unaffected.
- State polling invalidates only that worker's weather tile cache on a weather
  generation change.

The pipeline and render containers must share `cache_dir`, just as they do for
the current memmaps. No network client or decoder is created in render-only
workers.

## Concrete code changes

### New files

- `src/librewxr/data/weather_fields.py`
  - `WeatherFieldSpec`, registry, `WeatherFieldGeneration`,
    `GlobalWeatherFieldStore`, scale/offset encode/decode, state round-trip.
- `src/librewxr/sources/world/ifs/openmeteo_reader.py`
  - manifest parsing, run/path construction, one-open-per-valid-time component
    reads, shared O1280 regridding.
- `src/librewxr/sources/world/ifs/weather_fields.py`
  - time-window selection, candidate staging/validation, generation publisher,
    failure status.
- `src/librewxr/tiles/weather_field_renderer.py`
  - compute/present split and wind vector overlay.
- `src/librewxr/tiles/weather_field_palettes.py`
  - field LUTs, display ranges, and legend metadata.
- `tests/test_weather_fields.py`
  - registry, quantization, sampling, time interpolation, transaction/fallback,
    persistence.
- `tests/test_weather_fields_renderer.py`
  - global scalar/vector rendering, dateline/poles, palettes and output formats.
- `tests/test_weather_fields_api.py`
  - catalogue, immutable paths, validation, ETag/cache headers, stale/503 cases.

### Existing files to modify

- `src/librewxr/sources/world/ifs/grid.py`
  - use the extracted manifest/path/regrid reader; preserve all public
    precipitation methods and dBZ output.
- `src/librewxr/sources/world/ifs/__init__.py`
  - expose a weather-field provider in addition to the current NWP provider.
- `src/librewxr/sources/_base.py`
  - add `WeatherFieldSource` and `WeatherFieldContribution` without changing
    `NWPGrid`.
- `src/librewxr/sources/__init__.py`
  - discover and collect `weather_field_provider(settings, cache_dir)`.
- `src/librewxr/data/fetcher.py`
  - accept/fetch/close weather-field contributions under an explicit
    concurrency and status policy.
- `src/librewxr/data_pipeline.py`
  - construct the contribution and add `weather_fields` to the shared stores.
- `src/librewxr/main.py`
  - wire both modes, resurrection/state polling, dedicated cache, targeted
    invalidation, and memory monitoring.
- `src/librewxr/data/master_state.py`
  - tests/documentation for immutable-generation state and retain-old-on-apply
    failure; a generic resurrection helper may live here or in `main.py`.
- `src/librewxr/api/models.py`
  - additive weather catalogue response models.
- `src/librewxr/api/routes.py`
  - additive catalogue/tile routes, health/memory blocks, and route globals;
    leave Rain Viewer response construction untouched.
- `src/librewxr/tiles/cache.py`
  - likely no data-structure change; reuse it as a second instance. Add only a
    generic namespace invalidation helper if targeted clearing cannot stay in
    the route/main wiring.
- `src/librewxr/config.py`, `.env.example`, and
  `docs/configuration-reference.md`
  - enable flag, field list, history/horizon, weather tile cache budget, and
    operational sizing.
- `README.md`
  - document the additive API, source attribution, and RAM/disk impact.
- `tests/test_ecmwf_grid.py`
  - prove the precipitation adapter is byte-compatible after reader extraction
    and still skips T+0.
- `tests/test_api.py`, `tests/test_master_state.py`,
  `tests/test_poll_state.py`, `tests/test_data_pipeline.py`,
  `tests/test_renderer.py`, and the cache tests in `tests/test_fetcher.py`
  - regression and integration coverage described below.

`data/nwp_source.py`, the existing precipitation functions in
`tiles/renderer.py`, and `tiles/png_palette.py` should remain behaviourally
unchanged. If implementation pressure suggests changing their public
contracts, stop and revise this plan first.

## Migration and operational risks

1. **Resource growth.** Five int16 global components cost about 65 MB per
   native time before filesystem/page-cache effects. A 48-hour horizon is
   several GB on disk. Make horizon and enabled fields explicit, measure actual
   sparse/compressed alternatives, and document sizing before default enable.
2. **Regrid CPU and transient RAM.** Reading all fields at many valid times can
   exceed the current pipeline budget if float arrays coexist. Process one
   valid time at a time, release O1280/regridded temporaries promptly, and use a
   dedicated semaphore rather than multiplying IFS work across threads.
3. **Upstream schema drift.** `latest.json.variables` is advisory and T+0
   proves children vary by valid time. Validate each actual `.om` object and
   fail the candidate cleanly.
4. **Run/time identity.** The same valid timestamp can appear in successive
   runs with different values. Generation-qualified URLs and cache keys are
   mandatory.
5. **Atomicity across filesystems.** Atomic rename is guaranteed only within
   one filesystem. Staging and generation directories must be under the same
   `cache_dir`; never stage in `/tmp` and move across mounts.
6. **Worker races and cleanup.** A slow worker may still map the prior
   generation. Retain two generations plus a grace window; do not unlink the
   previous generation immediately after publishing `state.json`.
7. **Mixed deployment versions.** Old renderers ignore the new store. New
   renderers starting from old state must keep an empty recoverable store and
   self-heal when the pipeline is upgraded. No state-version bump is required
   for an additive store, but mixed-version tests are required.
8. **Palette interpretation.** Fixed global display ranges can hide extremes;
   per-tile auto-ranging makes animation flicker and legends meaningless. Use
   documented fixed palettes/ranges first and version named styles when they
   change.
9. **Wind conventions.** U/V describe motion toward east/north, while
   meteorological direction describes where wind comes from. Pin conversion
   tests with cardinal vectors.
10. **Pressure units.** The `.om` child returns Pa although Open-Meteo's public
    field is commonly presented as hPa. Convert exactly once at ingest and
    record canonical units in the manifest.
11. **Licence/attribution.** Preserve “ECMWF IFS, provided by Open-Meteo.com
    (CC-BY-4.0)” in the catalogue and documentation. Confirm whether tile-level
    UI attribution is required before public deployment.
12. **Existing cache invalidation.** Do not fold weather generation into the
    current NWP signature that clears radar tiles. Separate cache instances
    make this straightforward.

## Test plan

### Source and store unit tests

- Parse a representative `latest.json` with all target names and reject one
  missing component.
- Prove T+0 is selected for weather fields while precipitation continues to
  start at T+1.
- Mock one `.om` group containing precipitation plus all weather components and
  assert it is opened once per valid time.
- Verify pressure Pa→hPa, temperature/dew point °C, and U/V m/s conversions.
- Round-trip min/max/typical values through each int16 scale/offset codec;
  preserve the missing sentinel.
- Verify global shape/orientation, longitude roll, dateline wrapping, pole
  clamping, nearest and bilinear spatial sampling.
- Verify scalar linear interpolation and U/V component interpolation, including
  a 350°→10° equivalent wind case.
- Reject a candidate when any file, component, valid time, shape, or checksum
  is missing; assert the previous generation object and samples are unchanged.
- Simulate a crash in every staging phase and prove restart loads the last
  complete generation and removes only orphan staging files.
- Round-trip `__getstate__`/`__setstate__`; make a replacement file disappear
  during apply and prove the consumer retains the previous generation.

### API and renderer tests

- Catalogue schema, field units/kinds/times, provenance, stale state, and
  generation-qualified paths.
- No-generation 503 versus unknown generation/field/time 404.
- World tiles at the dateline, both poles, oceans, and representative land
  areas are valid and non-empty for complete scalar data.
- Temperature/dew-point/pressure LUT boundaries and missing alpha.
- Wind speed colours plus cardinal arrow directions.
- PNG/WebP decoding, deterministic PNG, ETag/304, immutable cache headers, and
  separate cache-key variants.
- A weather cache fill/eviction does not change the radar `TileCache` size or
  entries.

### Single/multi integration tests

- Single mode wires the store only when enabled and continues to serve all
  existing endpoints when disabled/unloaded.
- Pipeline snapshot includes one complete immutable weather generation.
- Render-only boot opens it, a later state poll swaps generations, and an
  in-flight sample on the old object completes.
- A corrupt next-generation state refresh logs degradation and continues to
  serve the old generation.
- New worker + legacy `state.json` self-heals after the next modern snapshot.
- Weather generation changes invalidate only the weather cache; existing radar
  NWP invalidation tests remain unchanged.

### Regression suite

Run at minimum:

```bash
.venv/bin/pytest tests/test_ecmwf_grid.py tests/test_ecmwf_interpolation.py
.venv/bin/pytest tests/test_api.py tests/test_renderer.py tests/test_fetcher.py
.venv/bin/pytest tests/test_master_state.py tests/test_poll_state.py tests/test_data_pipeline.py
.venv/bin/pytest tests/test_weather_fields.py tests/test_weather_fields_renderer.py tests/test_weather_fields_api.py
.venv/bin/pytest
```

Network integration probes should be explicit/manual and never part of the
normal unit suite. One probe should range-read live `.om` metadata and assert
the required child names without calling the point forecast API.

## Commit plan

Keep reviewable commits with green tests at every boundary:

1. **Extract shared Open-Meteo IFS run reader**
   - move manifest/path/regrid mechanics out of `ECMWFGrid`;
   - prove precipitation output and tests are unchanged.
2. **Add typed global weather field store**
   - specs, codecs, immutable generations, persistence, sampling, and temporal
     interpolation; no HTTP routes yet.
3. **Ingest global weather fields from IFS OM files**
   - provider discovery, staged transaction, validation, fetcher integration,
     failure fallback, and configuration.
4. **Share weather field generations in single and multi mode**
   - pipeline/main/master-state wiring, resurrection, health, cleanup/grace,
     and cross-process tests.
5. **Render and serve global weather field tiles**
   - palettes, compute/present renderer, dedicated cache, catalogue/tile API,
     ETags and cache headers.
6. **Document global weather fields operations**
   - README, configuration reference, `.env.example`, attribution, resource
     sizing, and deployment smoke instructions.

Do not combine changes to existing Rain Viewer output with these commits.

## Remaining unknowns to resolve before implementation

- Measure real S3 byte-range traffic and decode/regrid wall time when five
  components are read from one `.om`; metadata confirms co-location but not the
  production bandwidth/CPU cost.
- Benchmark 24 h versus 48 h default horizons against the pipeline's 12 GB
  multi-mode limit and typical single-mode hosts.
- Decide final fixed palettes/ranges and whether pressure contours belong in
  v1 or a later named style.
- Confirm the desired public forecast horizon and whether clients need
  interpolated catalogue timestamps or only native valid times.
- Confirm attribution placement requirements for downstream map UIs.
- Test atomic directory rename and retained-generation cleanup on the supported
  Docker named-volume, bind-mount, NFS, macOS, and Windows filesystem shapes.
- Check whether Open-Meteo offers a documented stability/version contract for
  `data_spatial` child names; the live manifest and open-data README establish
  current behaviour, not an immutable schema guarantee.

## Short conclusion

- **Fields actually available now:** `temperature_2m`, `dew_point_2m`,
  `pressure_msl`, `wind_u_component_10m`, and
  `wind_v_component_10m` are all present in the current Open-Meteo IFS
  `latest.json` and physically co-located with `precipitation` in T+1 `.om`
  files. All five instantaneous fields are also present at T+0, where
  accumulated precipitation is absent.
- **Chosen source:** the existing anonymous Open-Meteo
  `data_spatial/ecmwf_ifs` `.om` dataset. No point forecast API, separate
  dataset path, or direct ECMWF GRIB is needed for v1.
- **Unknowns:** production bandwidth/CPU/RAM, final horizon and palettes,
  attribution placement, cross-filesystem cleanup behaviour, and the upstream
  schema-stability guarantee still require measurement or confirmation.
- **Next-stage files:** the new store/reader/loader/renderer/palette/test files
  listed above, plus targeted changes to IFS `grid.py` and provider discovery,
  `fetcher.py`, `data_pipeline.py`, `main.py`, `master_state.py`, API
  routes/models, configuration, and documentation. Existing precipitation
  `NWPSource`, radar renderer behaviour, and PNG palette encoding remain intact.
