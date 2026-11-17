# OPERA fallback and observation freshness — 17 November 2026

Implementation deployed and verified on VPS on 17 November 2026.
Release: `opera-freshness-20261117-460be5ad`. No commits created by this task.
Baseline: `0f500e0`. Changes are in the local LibreWXR working tree.

## Behavior

- OPERA tries Cloudferro and then the official ECMWF backup at each requested
  timestamp, before trying up to three older five-minute products. The backup
  includes its own bucket path. Transport failures, HTTP errors, malformed
  products and mismatched HDF5 timestamps permit failover. Attempts are bounded.
- OPERA returns decoded pixels with the HDF5 observation time. Carry-forward
  inherits that original time and cannot extend its two-fetch-interval budget
  by copying an already copied frame.
- Per-region times survive both disk-cache restoration and renderer snapshot
  reload. A real observation replacing identical pixels still updates the
  metadata and content version. Recovery retries are reconstructed after restart.
- Point API freshness follows the selected region's observation time. A new
  composite slot or another region's fresh data cannot refresh old OPERA data.
  Observed frames expose `observation_time`; animation `time` and offsets keep
  their existing slot meaning.
- Legacy snapshots/cache files without provenance retain unknown source time.
  They cannot seed carry-forward and are retried within the recovery window.
  Point API reports null time/age and `stale: true` until provenance is known.

The source protocol still accepts plain arrays from existing providers, whose
initial observation time remains the requested slot. The generic carry-forward
age limit applies to every region. This change does not infer the true product
time of other providers or change OPERA's nodata/undetect pixel encoding.

## Validation

Final full run in `/Users/igor/projects/external/LibreWXR`: **2137 passed**,
61 deprecation warnings, 80.84 seconds. `git diff --check` passed.

Run from the repository root using the existing project environment:

```sh
.venv/bin/python -m pytest -q
git diff --check
```

Regressions cover primary success, backup success after HTTP/transport/parse
errors, older-product timestamps, bounded total failure, repeated carry-forward,
both restart paths, recovery retries, identical-pixel provenance updates,
unknown legacy metadata, selected-region freshness through the HTTP response,
and source discovery import order. Tests use in-memory HDF5 and mocked HTTP.

## Deployment handoff

Pipeline and both renderers deployed at 05:09–05:11 UTC. Successive OPERA
observations advanced from 05:05 to 05:10 UTC with changed precipitation values.
The point API retained true observation time and reported age 371 s, stale=false.
Production browser checks passed for Vilnius, Kaunas and Klaipeda: precipitation
continues and the conflict warning is absent. London still has a real discrepancy:
fresh OPERA is dry while the earlier station report says rain.

Both local and production site smoke checks passed. At one publication boundary,
the previous observation exceeded 600 s and temporarily became unavailable until
the next snapshot arrived; it was not advertised as fresh. This UX gap is a
separate follow-up. Temporary verification forwards were closed.

Release log and rollback instructions:
`/Users/igor/projects/theweather/docs/backend/deployments/librewxr-opera-2026-11-17.md`.
VPS archive, manifests, exact configuration and rollback script:
`/home/deploy/librewxr-rollouts/opera-freshness-20261117-460be5ad/`.
