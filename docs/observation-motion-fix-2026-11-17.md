# Motion from distinct observations — 17 November 2026

Deployed and browser-verified on VPS on 17 November 2026. No new commits.
Release: `animation-20261117-91f64085`, preserving the preceding OPERA
fallback/provenance release as its image base.

`data/nowcast.py` now selects distinct per-region observation times instead
of blindly comparing the last two fetch slots. Repeated publications no longer
zero optical flow. Displacement is normalized by the source-time interval;
coarse native cadence remains supported. Unknown provenance, stale latest
observations and gaps over two native/fetch intervals cannot establish motion.
The historical repeated pair keeps its own interpolation rather than borrowing
the forecast flow.

Full test suite: 2147 passed, 61 existing deprecation warnings, 92.28 s.
Regression coverage is in `tests/test_nowcast_observation_pairs.py`; existing
native cadence tests remain green. `git diff --check` passed.

Production snapshot reproduced the failure: last slots 05:20 and 05:25 UTC
both had observation 05:20; stored OPERA flow mean absolute component was
4.20e-10 pixels/step. Offline replay on reduced actual scans gave old-pair flow
9.55e-11 versus corrected 0.2143 pixels/step, with 151590 pixels changed in the
next forecast. No production pipeline state was modified for this check.

The web repo separately clamps negative crossfade progress, confirmed by a
browser MapLibre error and a regression test. Full diagnosis and release handoff:
`/Users/igor/projects/theweather/docs/backend/librewxr-animation-2026-11-17.md`.

Deployment log and rollback commands:
`/Users/igor/projects/theweather/docs/backend/deployments/librewxr-animation-2026-11-17.md`.
VPS artifacts: `/home/deploy/librewxr-rollouts/animation-20261117-91f64085/`.
Production flow became nonzero; the first four forecast tiles differ from the
current observation. Browser drag and full playback visibly move precipitation,
with no raster-opacity errors. Production smoke and post-cycle freshness passed.
