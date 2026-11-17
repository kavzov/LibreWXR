# Worker startup and memory diagnostics backport — 2026-11-17

Local backport on top of `d1cf47e`, validated before deployment. Deployed later on 2026-11-17 as recorded below.

## Upstream sources

- [35ab753](https://github.com/JoshuaKimsey/LibreWXR/commit/35ab75393e301e2db4fcb563b2d5e740df367c5a): resolve an unset bind host before uvicorn's multi-worker socket bind. Explicit addresses and single-worker behavior remain unchanged; multiple workers use `::`, falling back to `0.0.0.0` when IPv6 sockets are unavailable.
- [2891820](https://github.com/JoshuaKimsey/LibreWXR/commit/2891820990118c247e3e4ce3463772deca2d0f0e): PSS, per-worker details and a separate pipeline pulse. Only the diagnostics and their tests were backported; no dashboard or Prometheus dependency.
- [8c281b9](https://github.com/JoshuaKimsey/LibreWXR/commit/8c281b936c6a93c18afc6f8740618bd302cbdef0): expose the kernel's container usage as `current_mb`, without double-counting shared memory through `anon + file + shmem`.

## Health contract

Existing `/health` fields remain. The additive `cluster` fields are:

- `workers`: `worker_id`, `pid`, `role`, `rss_mb`, `pss_mb`, `tile_cache_mb`, `age_s` per render worker. The live worker replaces its disk pulse. Identity remains container + PID, so identical PIDs in separate render containers remain distinct.
- `workers_pss_mb`: sum/min/max of available render-worker PSS values, or `null` when none are available. A partially reporting set is not a complete container total. RSS and PSS are alternative measurements, not quantities to add together.
- `pipeline`: `reporting: false` when its pulse is absent or older than 60 seconds; otherwise PID, RSS, PSS and pulse age. Pipeline memory is not included in render-worker sums.
- `memory.container.current_mb`: `memory.current` on cgroup v2 or `memory.usage_in_bytes` on v1. It includes clean page cache and kernel memory. When that file cannot be read but the stat file can, the existing decision metric is the fallback. The container block describes the responding worker's container, not all containers on the VPS.

On systems without `/proc/self/smaps_rollup`, PSS is `null`; health and pulse publication continue. Each process publishes a jittered pulse approximately every 15 seconds. Collection and publication run off the event loop, as does the health endpoint's cluster scan, since reading PSS can walk process page tables. The pipeline loop stops before pipeline resource cleanup.

## Fork behavior preserved

Cache eviction still uses `anon + shmem` (v2), `rss + shmem` (v1), or the existing fallback. `current_mb` is diagnostic only and cannot by itself trigger eviction. Base thresholds remain 80/85/90%, with existing per-worker jitter, two-check hysteresis and the 60-second cooldown. Warning latches and recovery behavior are unchanged.

The boot-frame restore helper still releases restored frame references. Renderer snapshot ownership, alert-version comparison, coordinate-cache counters (`heap_held`), render queue diagnostics, OPERA observation times and nowcast motion fixes remain intact.

## Validation

The focused upstream tests cover bind resolution (including a real uvicorn socket), PSS parsing, missing/stale pipeline pulses and cgroup v1/v2 accounting. Fork regressions cover equal PIDs in distinct containers, replacing the live worker's old pulse, high clean-page-cache usage without eviction, exact thresholds and the default full-clear cooldown. Both pulse loops are checked for collection outside the event loop, recovery after a transient metrics error and orderly shutdown.

Validation on macOS, project `.venv`, Python 3.13.3:

- `.venv/bin/python -m pytest -q`: **2179 passed**, 61 warnings, 79.90 s. Warnings concern dependency deprecations and existing test/runtime paths; no failures.
- `git diff --check`: passed.
- AST comparison against `HEAD` confirms that all `MemoryMonitor` methods except the diagnostic `cgroup_memory_mb` property are unchanged, as are threshold jitter, memory release and the boot-frame restore helper.
- Linux cgroup/PSS behavior is covered with deterministic fixtures; production runtime verification is deferred. No deployment, restart, commit or push was performed.

## Deployment

Deployed on 2026-11-17 as `memory-health-20261117-0174f2e4`, sequentially updating pipeline and both renderers. The older pipeline image also required the already-tested `data/master_state.py` from `d1cf47e` to satisfy the updated `main.py` import; the isolated image check caught this before production switching.

At 06:41 UTC, both renderers and the router returned healthy responses with four distinct workers (two per container), positive PSS values and a fresh, advancing pipeline pulse. Container `current_mb` matched contemporaneous kernel readings within 1 MiB after startup settled. No restarts, OOM or application errors were observed. Production site smoke passed. Explicit production bind addresses, dependencies, runtime configuration and all other containers stayed unchanged.

Rollback to `animation-20261117-91f64085` is prepared in `/home/deploy/librewxr-rollouts/memory-health-20261117-0174f2e4/`: tagged old images, a verified `rollback-images.tar` archive, frozen Compose configurations and `release.py rollback <service>`. The release log in theweather is `docs/backend/deployments/librewxr-memory-health-2026-11-17.md`; a copy is stored alongside the VPS artifacts. No commit or push was performed during deployment.
