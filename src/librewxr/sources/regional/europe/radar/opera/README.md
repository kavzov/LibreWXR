# OPERA pan-European composite

CIRRUS pan-European radar composite operated by **EUMETNET / OPERA**,
published as ODIM HDF5 to Cloudferro S3 (`openradar-24h`), with the official
[ORD backup](https://eumetnet.github.io/openradardata-documentation/1-ORD-API-overview/#ord-backup)
at `https://object-store.os-api.cci1.ecmwf.int/openradar-24h-backup`.

## Coverage

| Region   | Footprint                                          |
| -------- | -------------------------------------------------- |
| `OPERA`  | Europe (Iceland–Turkey, S. Med–N. Scandinavia)     |

Native grid: 3800 × 4400 at 1 km on LAEA
(`+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000`).  In-repo
bbox is trimmed to the actual radar network extent, not the full LAEA
grid.

This is a genuinely multi-country source (~30 European countries) so
the package lives directly under `sources/regional/europe/radar/` with
no per-country directory.

## Cadence & latency

- Native cadence: **5 min**.
- Files publish with ~5–10 min delay. At each timestamp the source tries
  the primary and then the backup before walking back up to 3 older slots
  (`_MAX_FALLBACK_STEPS = 3`). Missing files, server/transport errors and
  invalid products allow failover. There are at most eight URL attempts,
  each with the existing bounded transport retry policy.
- `RadarObservation` carries the actual `what.date`/`what.time` from the
  HDF5; products whose time differs from their requested key are rejected.
  Older-slot fallback never acquires the requested slot's observation time.
- Carry-forward inherits the original observation time and expires after
  two fetch intervals measured from that original, even across restarts.
  Both disk metadata and renderer snapshots persist per-region times.
  Legacy caches without provenance have unknown age, are not carried forward,
  and are refetched within the normal recovery window.
- Archive: **rolling 24 hours** at the S3 bucket — older frames are
  pruned upstream.

## Format quirks

- ODIM HDF5; reflectivity at `dataset1/data1/data`, attrs at
  `dataset1/data1/what` (`gain`, `offset`, `nodata`, `undetect`).
- OPERA CIRRUS uses `gain=1.0`, `offset=0.0` — raw values are dBZ
  directly.
- Sentinels: `nodata=-9999000.0` (no radar coverage), `undetect=
  -8888000.0` (coverage but below detection).  Both are mapped to
  uint8 0 — OPERA acts as a **gap-filler** that only contributes
  pixels with actual precipitation.  Clear-sky pixels fall through
  to ECMWF so OPERA's inconsistent "undetect" swaths over open water
  don't suppress ECMWF over the same area.

## License & attribution

EUMETNET OPERA data policy — **open, gratis, anonymous**.  Citation
block in the top-level `README.md` and `docs/coverage.md`.

## Stations

~155 operational European radars from the EUMETNET OPERA database.  The
full list lives in `stations.py` together with the 300 km C-band range
override; both feed `data/coverage.py` via the `RadarSourceContribution`.
Regenerate by downloading `Data/OPERA_RADARS_DB_<date>.json` from the
OPERA Database page and filtering on `status="1"`.
