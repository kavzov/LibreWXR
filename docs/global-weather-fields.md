# Global weather map layers

LibreWXR serves global model fields alongside, but separately from, its
Rain Viewer-compatible radar API. These layers use ECMWF IFS data republished
as bulk `.om` files by Open-Meteo; LibreWXR does not fan out requests to the
Open-Meteo point forecast API.

## Enable the layers

The layers are enabled by default when ECMWF IFS is enabled. A persistent cache
is strongly recommended because the default 48-hour window is several GiB:

```dotenv
LIBREWXR_ECMWF_ENABLED=true
LIBREWXR_WEATHER_FIELDS_ENABLED=true
LIBREWXR_WEATHER_FIELDS_FORECAST_HOURS=48
LIBREWXR_WEATHER_FIELDS_MAX_TIMESTEPS=0
LIBREWXR_WEATHER_FIELDS_FIELDS=temperature_2m,dewpoint_2m,relative_humidity_2m,pressure_msl,wind_speed_10m
LIBREWXR_CACHE_DIR=/data/cache
```

Set `LIBREWXR_WEATHER_FIELDS_ENABLED=false` to retain the pre-feature
precipitation, snow-mask, radar, and nowcast behaviour without downloading or
storing the extra scalar fields. The `/v2/weather/metadata.json` response then
contains no available fields/timestamps and weather tile requests return 503;
the Rain Viewer endpoints are unaffected.

## Compatibility contract

| Existing subsystem | Preserved behaviour |
|---|---|
| Rain Viewer metadata | `/public/weather-maps.json` keeps the v2 `version/generated/host/radar/satellite` schema; scalar fields are not inserted. |
| Radar tiles | `/v2/radar/...` still calls legacy `NWPChain.sample()` for uint8 dBZ precipitation and uses the same geometry/presentation path. |
| Nowcast | Model fill and NWP flow continue through the precipitation chain; generic scalar sampling does not replace it. |
| Satellite | GMGSI routes, state, renderer, and cache namespace are independent of weather fields. |
| Alerts | WMO/NWS fetching and the `/v2/alerts` contract are independent; multi workers consume alert snapshots as before. |
| Single mode | One process owns fetch, atomic weather publication, rendering, and targeted cache invalidation. |
| Multi mode | The pipeline is the only downloader; render-only workers read local `state.json` and read-only memmaps. |
| Old cache/state | Missing manifests are tolerated, the last-known-good manifest falls back to legacy `state.json`, and v1 implicit precipitation/snow frame descriptors remain readable. |
| Fields disabled | IFS precipitation/snow remains enabled; scalar field storage/API availability disappears without changing existing endpoints. |

## Fields, source variables, and units

| Public field | Open-Meteo `.om` child | API unit | Storage | Notes |
|---|---|---|---|---|
| `temperature_2m` | `temperature_2m` | °C | `int16`, 0.1 °C | Native IFS field |
| `dewpoint_2m` | `dew_point_2m` | °C | `int16`, 0.1 °C | Public spelling deliberately omits the extra underscore |
| `relative_humidity_2m` | derived from temperature + dew point | % | not stored | Magnus formula, clipped to 0–100% |
| `pressure_msl` | `pressure_msl` | hPa | `uint16`, 0.1 hPa | Source values are converted from Pa |
| `wind_speed_10m` | derived from `wind_u_component_10m` + `wind_v_component_10m` | m/s | components are `int16`, 0.1 m/s | Speed is derived after spatial/time sampling |

The internal U/V components are generic weather fields but are not public
raster endpoints in this API version. Derived fields are calculated for the
requested tile; LibreWXR never materialises a full global humidity or wind
speed grid.

## Metadata and tile API

Fetch metadata first:

```http
GET /v2/weather/metadata.json
```

Abbreviated response:

```json
{
  "active_model_run": "2026-08-06T06:00:00Z",
  "generated": 1786017600,
  "stale": false,
  "attribution": "ECMWF IFS data via Open-Meteo",
  "fields": [
    {
      "id": "temperature_2m",
      "display_name": "Temperature at 2 m",
      "unit": "°C",
      "palette_ids": ["temperature"]
    }
  ],
  "available_timestamps": [1785996000, 1785999600],
  "default_timestamp": 1785996000,
  "palette_ids": ["temperature", "dewpoint", "humidity", "pressure", "wind_speed"],
  "palettes": [
    {
      "id": "temperature",
      "display_name": "Air temperature",
      "unit": "°C",
      "minimum": -50.0,
      "maximum": 50.0,
      "below_color": "#19094f",
      "above_color": "#4b0828",
      "nodata_color": "#00000000",
      "opacity": 1.0,
      "stops": [
        {"value": -50.0, "color": "#351a87"},
        {"value": 0.0, "color": "#d7f4f1"},
        {"value": 50.0, "color": "#7f143c"}
      ]
    }
  ],
  "tile_url_template": "http://localhost:8080/v2/weather/{field}/{timestamp}/{size}/{z}/{x}/{y}/{palette}.{ext}",
  "point_url_template": "http://localhost:8080/v2/weather/{field}/{timestamp}/point.json?lat={lat}&lon={lon}",
  "sizes": [256, 512],
  "formats": ["png", "webp"],
  "min_zoom": 0,
  "max_zoom": 12
}
```

The response contains complete palette stops, so clients should build legends
from metadata instead of duplicating colours. A tile URL is:

```http
GET /v2/weather/{field}/{timestamp}/{size}/{z}/{x}/{y}/{palette}.{ext}
```

For example:

```text
http://localhost:8080/v2/weather/temperature_2m/1785996000/256/4/8/5/temperature.png
http://localhost:8080/v2/weather/relative_humidity_2m/1785996000/512/4/8/5/humidity.webp
http://localhost:8080/v2/weather/pressure_msl/1785996000/256/4/8/5/pressure.png
```

Use a timestamp listed by metadata. The renderer also accepts an intermediate
Unix timestamp within the advertised range and linearly interpolates the two
surrounding native valid times. Supported sizes are 256 and 512; formats are
PNG and WebP; zoom is limited by `LIBREWXR_MAX_ZOOM`.

For an exact physical value at one coordinate, use the point template from
metadata:

```http
GET /v2/weather/{field}/{timestamp}/point.json?lat={lat}&lon={lon}
```

The response includes the field, timestamp, requested coordinates, value,
unit, active model run, and stale flag. `value` is `null` for nodata. Sampling
uses the same bilinear spatial and temporal interpolation path as the tiles;
point responses use `Cache-Control: no-store` so an active model update can
revise the same valid timestamp immediately.

## Attribution

Applications displaying these layers should show the metadata attribution:

```text
ECMWF IFS data via Open-Meteo
```

Keep links to [ECMWF](https://www.ecmwf.int/) and
[Open-Meteo](https://open-meteo.com/) in an about/data-sources view. The tile
layer is model output, not an observed radar or station product.

## Time, publication, and stale behaviour

Continuous fields use bilinear spatial interpolation and linear interpolation
between native valid times. Sampling follows this order:

```text
tile coordinates → cached grid index plan → frame A samples → frame B samples
→ temporal interpolation → optional humidity/wind derivation → palette
```

No request interpolates a full global frame. Precipitation retains its older
nearest/native and optical-flow path, so radar and nowcast behaviour does not
change.

A new IFS run is staged into versioned per-run memmaps. Each file is flushed,
size-checked, and atomically renamed; the active manifest and in-memory frame
mapping change only after every configured field and valid time succeeds. If a
run is incomplete, corrupt, or upstream is unavailable, LibreWXR keeps serving
the last complete run and exposes `stale` plus `last_update_error` in
`/health`. Empty tiles are not substituted for a last-known-good run.

## Storage and memory planning

The regular global grid is 1801 × 3600 cells. The default native set stores
five two-byte components (temperature, dew point, pressure, U wind, V wind),
or 64,836,000 bytes (61.8 MiB) per valid time. With hourly IFS valid times over
the default window, allow approximately:

| Retention | Approximate weather-field disk use |
|---|---:|
| One valid time | 61.8 MiB |
| One 48-hour run (about 49–50 valid times) | 3.0 GiB |
| Active + previous complete run | 6.0 GiB |
| Publication safety headroom while staging a third run | up to about 9 GiB |

Precipitation/snow, radar, satellite, nowcast, and tile-cache storage are
additional. Exact use depends on the valid times actually published and any
`LIBREWXR_WEATHER_FIELDS_MAX_TIMESTEPS` cap. Restrict
`LIBREWXR_WEATHER_FIELDS_FIELDS` or the horizon on smaller installations.

The arrays are read-only NumPy memmaps after publication. Each renderer maps
the files into virtual address space, but does not copy the whole run into its
Python heap; resident memory is driven by the pages touched by active tile
requests and the operating-system page cache. `/health` reports bytes per
field and the active model run.

## Single and multi deployment

In `single` mode the application fetches, atomically publishes, and renders the
same store. A model publication invalidates NWP-dependent radar/weather cache
entries without clearing unrelated satellite and coverage entries.

In `multi` mode only `python -m librewxr.data_pipeline` downloads and writes
model data. Render-only workers reopen descriptors from the atomic
`state.json` snapshot as read-only memmaps, poll for changes, and switch the
whole weather generation together. They do not create an upstream filesystem
client or perform downloads. The active and immediately previous run
directories are protected from cleanup so workers and in-flight renders can
finish against their prior snapshot.

Pipeline and renderer containers must share the same `LIBREWXR_CACHE_DIR` and
use matching weather-field settings.

## CDN and browser caching

- Cache metadata briefly and refresh it about once per minute; its response
  currently advertises `Cache-Control: public, max-age=60`.
- Tile responses include an ETag and support `If-None-Match`.
- Tile responses advertise `public, max-age=21600` (six hours). The timestamp
  is immutable within one model generation, but a later model run can revise
  the forecast for the same valid timestamp, so do not configure a CDN to
  ignore origin TTL/revalidation indefinitely.
- The internal cache key includes field, model/grid/content version,
  timestamp, coordinates, size, palette, format, encoding settings, and
  renderer version.
- Prefer WebP or quantized PNG for bandwidth. Exact PNG8 remains lossless;
  `LIBREWXR_WEATHER_PNG_MODE=lossless` is available when palette quantization
  is unacceptable.

## Accuracy and interpretation

- IFS is a numerical forecast/analysis model. It is not a thermometer,
  barometer, anemometer, radar, or satellite observation at the rendered
  pixel.
- The source model is roughly 9 km and is regridded to a 0.1° regular grid.
  High zoom levels provide smoother presentation, not additional atmospheric
  resolution.
- Encoded precision is 0.1 °C, 0.1 hPa, and 0.1 m/s. Relative humidity is
  derived from encoded temperature/dew point and wind speed from encoded U/V,
  so their accuracy cannot exceed their dependencies.
- Linear time interpolation makes animation continuous but does not add
  forecast skill between model valid times.
- Palette tiles are visual products, not numeric rasters. Do not recover
  scientific values by reverse-mapping PNG/WebP colours; expose a dedicated
  numeric API in a future change if exact point values are required.
