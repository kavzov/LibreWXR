# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import json
import logging
import math
import os
import pathlib
import re
import time
from collections.abc import Awaitable, Callable

import psutil
from PIL import Image

from fastapi import APIRouter, HTTPException, Path, Query, Request, Response

from datetime import datetime

from librewxr.api.models import (
    AlertProperties,
    AlertsResponse,
    ColorScheme,
    GeoJSONFeature,
    RadarData,
    RadarAnimationData,
    RadarMotionData,
    RadarPointNowcastResponse,
    RadarTimestamp,
    SatelliteData,
    WeatherFieldInfo,
    WeatherMapsResponse,
    WeatherMetadataResponse,
    WeatherPaletteInfo,
    WeatherPaletteStop,
    WeatherPointResponse,
)
from librewxr.api.conditional import compute_etag, conditional_response
from librewxr.colors.schemes import SCHEME_NAMES
from librewxr.colors.weather_palettes import (
    PUBLIC_WEATHER_FIELDS,
    WEATHER_PALETTES,
    palettes_for_field,
)
from librewxr.config import settings
from librewxr.data.pagecache import coord_pagecache_prime_stats
from librewxr.data.point_nowcast import build_point_nowcast
from librewxr.data.store import FrameStore
from librewxr.data.weather_fields import WeatherField, field_spec
from librewxr.data.worker_pulse import read_worker_pulses, worker_identity
from librewxr.mcp.discovery import build_ai_catalog
from librewxr.memory import detect_memory_limit_mb
from librewxr.tiles import window
from librewxr.tiles.cache import CachedRender, TileCache
from librewxr.tiles.coordinates import (
    coord_cache_bytes,
    coord_cache_stats,
    window_origin,
)
from librewxr.tiles.renderer import (
    _encode_image,
    _transparent_tile,
    compute_coverage_rgba,
    compute_tile_geometry,
    present_tile,
    render_coverage_tile,
    TileGeometry,
    transparent_fast_path_label,
)
from librewxr.tiles.motion_renderer import (
    MOTION_ENCODING,
    MOTION_RENDERER_VERSION,
    MOTION_VECTOR_OFFSET,
    MOTION_VECTOR_SCALE,
    render_motion_tile,
)
from librewxr.tiles.request_tracker import RENDER_STAGE_NAMES, TileRequestTracker
from librewxr.tiles.render_queue import BoundedRenderQueue
from librewxr.tiles.satellite_renderer import (
    render_gmgsi_composite_tile,
    render_gmgsi_tile,
)
from librewxr.tiles.weather_renderer import (
    WEATHER_RENDERER_VERSION,
    render_scalar_weather_tile,
    sample_scalar_weather_point,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# These get set by main.py during startup
frame_store: FrameStore | None = None
tile_cache: TileCache | None = None
# All NWP grids live in a single dict keyed by slug
# (``hrrr_grid``, ``arome_antilles_grid``, ``ecmwf_grid``, etc.) —
# generated from ``NWPContribution.name`` via ``nwp_grid_slug``.  The
# ``/health`` endpoint iterates this dict so adding a new NWP source
# requires no edits here.  ``ecmwf_grid`` is also bound as an attribute
# below for the radar tile arrow path that still treats IFS specially.
nwp_grids: dict[str, object] = {}
ecmwf_grid = None  # ECMWFGrid | None — special-cased by /v2/radar arrows
nwp_chain = None  # NWPChain | None
precip_mask = None  # PrecipMaskStore | None — set by main.py (multi mode only)
# GMGSI satellite sources keyed by slug (gmgsi_lw_grid, gmgsi_vis_grid).
# Routes index by slug so the /health endpoint and tile dispatcher
# auto-pick up new channels without per-source plumbing.
satellite_grids: dict[str, object] = {}
tile_warmer = None  # TileWarmer | None
nowcast_store = None  # NowcastStore | None
storm_cell_store = None  # StormCellStore | None
radar_cache = None  # RadarFrameCache | None
radar_fetcher = None  # RadarFetcher | None
tile_request_tracker: TileRequestTracker | None = None
start_time: float = 0.0
enabled_regions: list[str] | None = None
# Memory monitor — set by main.py in both lifespans.  Provides the cgroup
# anon/file/shmem split for the /health ``cluster`` section.
memory_monitor = None  # MemoryMonitor | None

# Tile present pool - set by main.py.  Multi-mode render workers get a
# dedicated executor for the cheap ``present_tile`` tail (colorize/encode)
# so those jobs never queue behind long geometry computes on the shared
# default executor under a cold-tile burst.  Single mode leaves this None
# and the tile endpoints fall back to ``asyncio.to_thread`` (the loop
# default executor), byte-identical to the pre-split behaviour.
present_executor = None  # ThreadPoolExecutor | None

# Shared-store I/O pool - set by main.py in multi mode only.  Dedicated
# executor for shared-tile-store reads/publishes so they never queue
# behind geometry computes on the default executor.  Single mode leaves
# this None and the shared-store call sites fall back to
# ``asyncio.to_thread`` (the loop default executor).
io_executor = None  # ThreadPoolExecutor | None

# Multi-mode admission control in front of the compute executor.  The
# executor itself has an unbounded work queue; this object limits submitted
# jobs while excess request coroutines wait cheaply on the event loop.
render_queue: BoundedRenderQueue | None = None

# Shared on-disk encoded-tile store - set by main.py in multi mode only.
# A ``radar_tile`` hit skips frame fetch + geometry compute + present
# entirely; plain present misses publish their fresh encode for the other
# workers.
shared_tile_store = None  # SharedTileStore | None

# Holds fire-and-forget shared-store publish tasks so they can't be GC'd
# mid-flight; each task discards itself on completion.
_pending_shared_publishes: set = set()

# Memoized encoded bytes + ETag for fully transparent (ocean/clear-sky)
# tiles: (tile_size, ext) -> (tile_bytes, etag).  These are the dominant
# global request class; their bytes are process constants per
# (tile_size, ext).  settings.webp_quality is process-static so it is
# deliberately not part of the key (a quality change requires a restart
# anyway).
_TRANSPARENT_RENDER_MEMO: dict[tuple[int, str], tuple[bytes, str]] = {}

# Latest-timestamp TTL cache for the radar tile hot path:
# (monotonic time, timestamp list).  ``radar_tile`` only needs the latest
# frame to pick the Cache-Control ``max_age`` bucket (300 s vs 7200 s), so
# re-querying the store lock on every request is wasted contention; 5 s of
# staleness is immaterial next to those buckets.
_latest_ts_cache: tuple[float, list[int]] | None = None
_LATEST_TS_TTL = 5.0

# WMO alerts — set by main.py during startup
alerts_store = None  # AlertsStore | None
alerts_fetcher = None  # WMOAlertsFetcher | None
alerts_enabled: bool = False

# MCP server — set by main.py during startup (only when settings.mcp_enabled
# is True AND the [mcp] extra successfully imported + mounted).  ``mcp_mounted``
# distinguishes "config asked for MCP but the build/import failed" (False)
# from "MCP endpoint is live and answering" (True).
mcp_mounted: bool = False
mcp_path: str = "/mcp"
mcp_tools: list[str] = []

# Per-process cold-render singleflight. Render workers do not share an event
# loop or byte cache, so each worker owns its own in-flight task map.
_weather_tile_flights: dict[tuple, asyncio.Task[CachedRender]] = {}
_geometry_flights: dict[tuple, asyncio.Task] = {}


async def _singleflight(
    flights: dict[tuple, asyncio.Task],
    key: tuple,
    factory: Callable[[], Awaitable],
) -> tuple[object, bool]:
    """Return one shared task result and whether this caller created it."""

    task = flights.get(key)
    leader = task is None
    if task is None:
        task = asyncio.create_task(factory())
        flights[key] = task

        def _remove(done: asyncio.Task) -> None:
            if flights.get(key) is done:
                flights.pop(key, None)

        task.add_done_callback(_remove)
    # A cancelled client must not cancel the shared render needed by peers.
    return await asyncio.shield(task), leader


async def _weather_tile_singleflight(
    key: tuple,
    factory: Callable[[], Awaitable[CachedRender]],
) -> CachedRender:
    """Share one cold weather render among concurrent identical requests."""

    result, _leader = await _singleflight(_weather_tile_flights, key, factory)
    return result


def _nwp_grid_health_blocks() -> dict[str, dict]:
    """Build per-grid ``/health`` blocks for every entry in ``nwp_grids``.

    IFS reports a different shape (``reference_time`` + ``timesteps``)
    than the chain-source grids (``latest_run`` + ``frames``).  Detect
    by attribute presence rather than slug — keeps the shape stable if
    a future provider adopts either pattern.
    """
    blocks: dict[str, dict] = {}
    for slug, grid in nwp_grids.items():
        if grid is None:
            blocks[slug] = {"enabled": False, "loaded": False}
            continue
        if hasattr(grid, "reference_time") and hasattr(grid, "timestep_count"):
            block = {
                "loaded": getattr(grid, "data", None) is not None,
                "reference_time": grid.reference_time,
                "timesteps": grid.timestep_count,
            }
            health_status = getattr(grid, "health_status", None)
            if health_status is not None:
                block.update(health_status())
            blocks[slug] = block
        else:
            blocks[slug] = {
                "enabled": True,
                "loaded": grid.has_data(),
                "latest_run": grid.latest_run_iso,
                "frames": grid.frame_count,
            }
    return blocks


def _avg_ms(total_ns: int, count: int) -> float:
    """Mean latency in milliseconds from ns totals; 0.0 when empty."""
    if count == 0:
        return 0.0
    return round(total_ns / count / 1e6, 2)


def collect_worker_pulse() -> dict:
    """Compact per-process payload for the cluster worker-pulse files.

    Every field is derived from the module-level singletons with None
    guards — render-only mode leaves several unset (``radar_cache``,
    ``radar_fetcher``, ``alerts_fetcher``, ``tile_warmer``).  The payload
    is deliberately small (< 2 KB) so a /health scan of 16 tiny JSON
    files stays cheap.
    """
    payload = {
        "worker_id": worker_identity(),
        "pid": os.getpid(),
        "written_at": int(time.time()),
        "rss_bytes": psutil.Process().memory_info().rss,
    }

    if tile_cache is not None:
        payload["tile_cache"] = {
            "entries": tile_cache.size,
            "total_bytes": tile_cache.total_bytes,
            "max_bytes": tile_cache.max_bytes,
        }

    coord: dict = {"caches": {}}
    try:
        coord_stats = coord_cache_stats()
    except Exception:
        coord_stats = None
    if coord_stats is not None:
        for name, info in coord_stats.get("caches", {}).items():
            coord["caches"][name] = {
                "entries": info["entries"],
                "hits": info["hits"],
                "misses": info["misses"],
            }
        coord["store"] = None
        store_stats = coord_stats.get("store")
        if store_stats is not None:
            coord["store"] = {
                "hits": store_stats["hits"],
                "misses": store_stats["misses"],
                "publishes": store_stats["publishes"],
                "async_pending": store_stats.get("async_pending", 0),
                "async_skipped": store_stats.get("async_skipped", 0),
            }
    payload["coord"] = coord

    requests = {"enabled": False}
    if tile_request_tracker is not None:
        try:
            tracker_stats = tile_request_tracker.stats()
        except Exception:
            tracker_stats = None
        if tracker_stats is not None:
            requests = {
                "enabled": True,
                "total_requests": tracker_stats["total_requests"],
                "hot_tiles": tracker_stats["hot_tiles"],
                "fast_path_total": tracker_stats["fast_path"]["total"],
                "cache_hits": tracker_stats["cache"]["hits"],
                "cache_misses": tracker_stats["cache"]["misses"],
            }
    payload["requests"] = requests

    if render_queue is not None:
        payload["render_queue"] = render_queue.snapshot()

    # Tile-latency accumulators, additive across workers: ns totals and
    # stage counts.  Old pulses that predate these fields are tolerated
    # by the aggregator via .get(..., 0).
    if tile_request_tracker is not None:
        try:
            lat = tile_request_tracker.latency_snapshot()
        except Exception:
            lat = None
        if lat is not None:
            payload["tile_latency"] = {
                "request_ns_total": lat["request_ns_total"],
                "request_count": lat["request_count"],
                "compute_ns_total": lat["compute_ns_total"],
                "compute_count": lat["compute_count"],
                "present_ns_total": lat["present_ns_total"],
                "present_count": lat["present_count"],
                "stages": lat["stages"],
            }

    return payload


def _cluster_health_section() -> dict:
    """Aggregate the live worker pulses into the /health ``cluster`` block.

    Reads the pid-unique pulse files the worker pulse loops write under
    ``<cache_dir>/workers/`` (mtime-filtered, no locks — see
    ``librewxr.data.worker_pulse``), then unions in THIS process's live
    payload by pid so a worker reports even before its first pulse write
    lands on disk.  Per-process counters (RSS, tile-cache bytes, tracker
    counts, coord-cache hits) are summed across workers; the coord store
    ``entries``/``bytes`` describe the single global on-disk store and
    come from this worker's live stats instead.

    Every read here is a tiny file scan; the caller wraps this in
    try/except so a scan failure degrades the section to None rather
    than breaking /health.
    """
    cache_dir = (
        pathlib.Path(settings.cache_dir) if settings.cache_dir else None
    )
    pulses = read_worker_pulses(cache_dir) if cache_dir is not None else []
    by_worker: dict[str, dict] = {}
    for pulse in pulses:
        if not isinstance(pulse, dict) or not isinstance(pulse.get("pid"), int):
            continue
        identity = pulse.get("worker_id")
        key = identity if isinstance(identity, str) else f"legacy-pid-{pulse['pid']}"
        by_worker[key] = pulse
    # The live payload is strictly fresher than any on-disk file this
    # process left behind, so it wins the pid-keyed union.
    live_pulse = collect_worker_pulse()
    by_worker[live_pulse["worker_id"]] = live_pulse
    pulses = list(by_worker.values())

    rss_values = [
        pulse["rss_bytes"] for pulse in pulses if pulse.get("rss_bytes")
    ]
    workers_rss_mb = {
        "sum": round(sum(rss_values) / (1024 * 1024), 1),
        "min": round(min(rss_values) / (1024 * 1024), 1),
        "max": round(max(rss_values) / (1024 * 1024), 1),
    }
    memory_block = {
        # cgroup split is only meaningful inside a container; None there.
        "container": (
            memory_monitor.cgroup_memory_mb if memory_monitor is not None else None
        ),
        "workers_rss_mb": workers_rss_mb,
    }

    tile_entries = sum(
        pulse["tile_cache"]["entries"] for pulse in pulses if pulse.get("tile_cache")
    )
    tile_bytes = sum(
        pulse["tile_cache"]["total_bytes"] for pulse in pulses if pulse.get("tile_cache")
    )
    tile_cache_block = {
        "entries": tile_entries,
        "used_mb": round(tile_bytes / (1024 * 1024), 1),
    }

    # Per-cache counters sum across workers; hit_ratio is recomputed from
    # the SUMS (a per-worker ratio averaged arithmetically would weight
    # idle workers as strongly as busy ones).
    cache_sums: dict[str, dict] = {}
    for pulse in pulses:
        for name, info in pulse.get("coord", {}).get("caches", {}).items():
            agg = cache_sums.setdefault(
                name, {"entries": 0, "hits": 0, "misses": 0},
            )
            agg["entries"] += info["entries"]
            agg["hits"] += info["hits"]
            agg["misses"] += info["misses"]
    for name, agg in cache_sums.items():
        total = agg["hits"] + agg["misses"]
        agg["hit_ratio"] = round(agg["hits"] / total, 3) if total else None
    coord_block = {"caches": cache_sums, "store": None}

    # Shared store: hits/misses/publishes are per-process counters and sum
    # across workers, but entries/bytes are a scan of the ONE global on-disk
    # store — every worker sees the same values, so summing would over-count.
    # They come from this worker's live stats instead.
    store_sums = {
        "hits": 0,
        "misses": 0,
        "publishes": 0,
        "async_pending": 0,
        "async_skipped": 0,
    }
    for pulse in pulses:
        store_stats = pulse.get("coord", {}).get("store")
        if store_stats:
            for key in store_sums:
                store_sums[key] += store_stats.get(key, 0)
    try:
        live_store = coord_cache_stats().get("store")
    except Exception:
        live_store = None
    if live_store is not None:
        coord_block["store"] = {
            **store_sums,
            "entries": live_store["entries"],
            "bytes": live_store["bytes"],
            "budget_bytes": live_store.get("budget_bytes", 0),
            "over_budget": live_store.get("over_budget", False),
        }

    # Tracked tile counts: hot_tiles is summed and can double-count a tile
    # that several workers all served — it's a cross-worker activity proxy,
    # not a distinct-tile count.  hit_rate is recomputed from the summed
    # hits/misses (mirroring the per-worker format: 0.0 when idle).
    requests_block = {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "fast_path_total": 0,
        "hot_tiles": 0,
    }
    for pulse in pulses:
        req = pulse.get("requests") or {}
        if not req.get("enabled"):
            continue
        for key in requests_block:
            requests_block[key] += req.get(key, 0)
    hits = requests_block["cache_hits"]
    misses = requests_block["cache_misses"]
    requests_block["hit_rate"] = (
        hits / (hits + misses) if (hits + misses) > 0 else 0.0
    )

    # Tile-latency sums: additive ns totals/counts across workers; the
    # cluster-wide averages are recomputed from the SUMS (mirroring the
    # hit_rate recomputation above — an arithmetic mean of per-worker
    # averages would weight idle workers as strongly as busy ones).
    # Pulses written before this field existed are tolerated via
    # .get(..., 0).
    lat_sums = {
        "request_ns_total": 0,
        "request_count": 0,
        "compute_ns_total": 0,
        "compute_count": 0,
        "present_ns_total": 0,
        "present_count": 0,
    }
    stage_sums = {
        name: {"ns_total": 0, "count": 0}
        for name in RENDER_STAGE_NAMES
    }
    for pulse in pulses:
        lat = pulse.get("tile_latency") or {}
        for key in lat_sums:
            lat_sums[key] += lat.get(key, 0)
        for name, stage_sum in stage_sums.items():
            stage = (lat.get("stages") or {}).get(name) or {}
            stage_sum["ns_total"] += stage.get("ns_total", 0)
            stage_sum["count"] += stage.get("count", 0)
    tile_latency_block = {
        "avg_request_ms": _avg_ms(
            lat_sums["request_ns_total"], lat_sums["request_count"],
        ),
        "avg_compute_ms": _avg_ms(
            lat_sums["compute_ns_total"], lat_sums["compute_count"],
        ),
        "avg_present_ms": _avg_ms(
            lat_sums["present_ns_total"], lat_sums["present_count"],
        ),
        "stages": {
            name: {
                "avg_ms": _avg_ms(stage["ns_total"], stage["count"]),
                "count": stage["count"],
            }
            for name, stage in stage_sums.items()
        },
    }

    queue_keys = (
        "worker_slots", "queue_slots", "capacity", "inflight",
        "executor_queued", "waiting", "peak_waiting", "admitted_total",
    )
    render_queue_block = {key: 0 for key in queue_keys}
    for pulse in pulses:
        queue = pulse.get("render_queue") or {}
        for key in queue_keys:
            render_queue_block[key] += queue.get(key, 0)

    return {
        "workers_reporting": len(pulses),
        "memory": memory_block,
        "tile_cache": tile_cache_block,
        "coord": coord_block,
        "requests": requests_block,
        "tile_latency": tile_latency_block,
        "render_queue": render_queue_block,
    }


@router.get("/.well-known/ai-catalog.json")
async def ai_catalog() -> Response:
    """AI Catalog (proposal) entry pointing at the MCP server card.

    Self-description directory entry that resolves to the SEP-2127
    (draft) server card at ``<mcp_path>/server-card``.  Draft proposal,
    not a ratified standard.  404s when MCP is disabled by config or the
    HTTP transport failed to mount (``mcp_mounted`` False).  CORS is
    handled by the parent app's CORSMiddleware.
    """
    if not settings.mcp_enabled or not mcp_mounted:
        raise HTTPException(status_code=404, detail="MCP not available")
    return Response(
        content=json.dumps(build_ai_catalog()),
        media_type="application/ai-catalog+json",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/health")
async def health():
    """Health and status endpoint."""
    now = int(time.time())
    uptime = now - int(start_time)
    mem_limit_mb = detect_memory_limit_mb(settings.memory_limit_mb)
    rss_bytes = psutil.Process().memory_info().rss
    rss_mb = rss_bytes / (1024 * 1024)
    ram_usage = round(rss_mb / mem_limit_mb * 100, 1)
    frame_count = await frame_store.frame_count()
    retained_frame_count = await frame_store.retained_frame_count()
    timestamps = await frame_store.get_timestamps()
    latest_ts = max(timestamps) if timestamps else None
    oldest_ts = min(timestamps) if timestamps else None

    # Per-region frame counts catch silent regional failures: if OPERA
    # falls behind while USCOMP keeps fetching, the totals diverge here.
    region_keys = await frame_store.get_region_keys()
    per_region_counts: dict[str, int] = {}
    for names in region_keys.values():
        for name in names:
            per_region_counts[name] = per_region_counts.get(name, 0) + 1
    for name in (enabled_regions or []):
        per_region_counts.setdefault(name, 0)

    # Per-component memory breakdown.  Every NWP grid is iterated from
    # ``nwp_grids``; the per-slug byte counts are folded into both
    # ``tracked_bytes`` and the ``breakdown`` dict below so adding a new
    # NWP source requires no edits here.
    radar_bytes = frame_store.data_bytes
    tile_cache_bytes = tile_cache.total_bytes
    nwp_bytes_by_slug: dict[str, int] = {
        slug: (grid.data_bytes if grid is not None else 0)
        for slug, grid in nwp_grids.items()
    }
    nowcast_bytes = nowcast_store.data_bytes if nowcast_store else 0
    satellite_bytes = sum(
        grid.data_bytes
        for grid in satellite_grids.values()
        if grid is not None
    )
    coord_stats = coord_cache_stats()
    store_stats = coord_stats.get("store")
    if store_stats is not None:
        # Store-backed: entries are shared read-only memmap pages, not private
        # heap - report the on-disk footprint separately, contribute 0 to RSS
        # reconciliation.
        coord_bytes = 0
    else:
        coord_bytes = coord_cache_bytes()
    tracked_bytes = (
        radar_bytes + tile_cache_bytes + sum(nwp_bytes_by_slug.values())
        + nowcast_bytes + satellite_bytes + coord_bytes
    )
    other_bytes = max(0, rss_bytes - tracked_bytes)

    breakdown = {
        "radar_frames_mb": round(radar_bytes / (1024 * 1024), 1),
        "tile_cache_mb": round(tile_cache_bytes / (1024 * 1024), 1),
    }
    for slug, nbytes in nwp_bytes_by_slug.items():
        breakdown[f"{slug}_mb"] = round(nbytes / (1024 * 1024), 1)
    breakdown.update({
        "nowcast_mb": round(nowcast_bytes / (1024 * 1024), 1),
        "satellite_mb": round(satellite_bytes / (1024 * 1024), 1),
        "coord_caches_mb": round(coord_bytes / (1024 * 1024), 1),
        "coord_store_mb": (
            round(store_stats["bytes"] / (1024 * 1024), 1)
            if store_stats else 0.0
        ),
        "coord_store_entries": store_stats["entries"] if store_stats else 0,
        "other_mb": round(other_bytes / (1024 * 1024), 1),
    })

    # Split the tile cache into its five entry kinds: satellite render
    # entries (``"sat"``-prefixed keys), geometry entries (int timestamp +
    # 6-element viewport key), present render entries (int timestamp +
    # 9-element viewport/visual key), overlay present entries (int
    # timestamp + 9-element viewport/visual key + 2-element style suffix,
    # nowcast frames only), and lat/lon-centered window present entries
    # (int timestamp + "win" + 9-element window/visual key).  Each kind is
    # reported with its own count and byte total.
    cache_kind_geometry = 0
    cache_kind_geometry_bytes = 0
    cache_kind_present = 0
    cache_kind_present_bytes = 0
    cache_kind_overlay = 0
    cache_kind_overlay_bytes = 0
    cache_kind_window = 0
    cache_kind_window_bytes = 0
    cache_kind_satellite = 0
    cache_kind_satellite_bytes = 0
    for key, size in tile_cache.entries():
        if key[0] == "sat":
            cache_kind_satellite += 1
            cache_kind_satellite_bytes += size
        elif key and isinstance(key[0], int) and len(key) == 7:
            cache_kind_geometry += 1
            cache_kind_geometry_bytes += size
        elif key and isinstance(key[0], int) and len(key) == 10:
            cache_kind_present += 1
            cache_kind_present_bytes += size
        elif key and isinstance(key[0], int) and len(key) == 12:
            cache_kind_overlay += 1
            cache_kind_overlay_bytes += size
        elif key and isinstance(key[0], int) and len(key) == 11 and key[1] == "win":
            cache_kind_window += 1
            cache_kind_window_bytes += size

    # Cluster-wide aggregation: lock-free scan of the tiny per-worker pulse
    # files under the shared cache dir, unioned with this worker's live
    # payload.  Degrades to None on any failure — never an exception.
    try:
        cluster = _cluster_health_section()
    except Exception:
        logger.exception("Failed to assemble cluster health section")
        cluster = None

    return {
        "status": "ok" if frame_count > 0 else "degraded",
        "uptime_seconds": uptime,
        "cluster": cluster,
        "memory": {
            "resident_mb": round(rss_mb, 1),
            "limit_mb": round(mem_limit_mb, 1),
            "usage_pct": ram_usage,
            "breakdown": breakdown,
        },
        "frames": {
            "count": frame_count,
            "max": settings.max_frames,
            "retained_count": retained_frame_count,
            "grace_max": settings.frame_grace_frames,
            "latest": latest_ts,
            "oldest": oldest_ts,
            "latest_age_seconds": now - latest_ts if latest_ts else None,
            "per_region": per_region_counts,
        },
        "tile_cache": {
            "entries": tile_cache.size,
            "used_mb": round(tile_cache.total_bytes / (1024 * 1024), 1),
            "max_mb": settings.tile_cache_mb,
            "geometry_entries": cache_kind_geometry,
            "geometry_bytes": cache_kind_geometry_bytes,
            "present_entries": cache_kind_present,
            "present_bytes": cache_kind_present_bytes,
            "overlay_entries": cache_kind_overlay,
            "overlay_bytes": cache_kind_overlay_bytes,
            "window_entries": cache_kind_window,
            "window_bytes": cache_kind_window_bytes,
            "satellite_entries": cache_kind_satellite,
            "satellite_bytes": cache_kind_satellite_bytes,
        },
        **_nwp_grid_health_blocks(),
        "nwp_chain": {
            "sources": [s.name for s in nwp_chain.sources] if nwp_chain else [],
        },
        "nowcast": {
            "enabled": settings.nowcast_enabled,
            "arrow_flow_enabled": settings.arrow_flow_enabled,
            "arrow_flow_target_dim": settings.arrow_flow_target_dim,
            "arrow_nwp_flow_resolution_deg": settings.arrow_nwp_flow_resolution_deg,
            "flows": len(await nowcast_store.get_flows() or {}) if nowcast_store else 0,
            "nwp_flow": await nowcast_store.get_nwp_flow() is not None if nowcast_store else False,
            "frames": await nowcast_store.get_timestamps() if nowcast_store else [],
            "count": len(await nowcast_store.get_timestamps()) if nowcast_store else 0,
            "animation_frames": (
                await nowcast_store.get_animation_timestamps()
                if nowcast_store else []
            ),
        },
        "satellite": {
            "enabled": settings.satellite_enabled,
            "channels": {
                slug: {
                    "loaded": grid is not None and bool(grid.timestamps),
                    "frames": len(grid.timestamps) if grid is not None else 0,
                    "latest": (
                        grid.timestamps[-1]
                        if grid is not None and grid.timestamps
                        else None
                    ),
                }
                for slug, grid in satellite_grids.items()
            },
        },
        "enabled_regions": enabled_regions or [],
        "sources": {
            "na_source": settings.na_source,
            "ca_source": settings.ca_source,
            # CACOMP MSC blending state: True/False once observed,
            # None if blending isn't configured for this region set.
            "cacomp_msc_blending": (
                radar_fetcher._cacomp_msc_available
                if radar_fetcher is not None
                and radar_fetcher._cacomp_msc_source is not None
                else None
            ),
        },
        "radar_cache": (
            {"enabled": True, **radar_cache.stats()}
            if radar_cache is not None
            else {"enabled": False}
        ),
        "coord_caches": coord_cache_stats(),
        "coord_pagecache_prime": (
            coord_pagecache_prime_stats(settings.cache_dir)
            if settings.cache_dir else None
        ),
        "render_queue": render_queue.snapshot() if render_queue is not None else None,
        "tile_requests": (
            {"enabled": True, **tile_request_tracker.stats()}
            if tile_request_tracker is not None
            else {"enabled": False}
        ),
        "alerts": {
            "enabled": alerts_enabled,
            "count": alerts_store.count if alerts_store is not None else 0,
            "last_updated": int(alerts_store.last_updated) if alerts_store is not None else 0,
            "ingest_ok": alerts_store.fetch_success if alerts_store is not None else False,
        } if alerts_enabled else {"enabled": False},
        "mcp": {
            "enabled": settings.mcp_enabled,
            "mounted": mcp_mounted,
            "path": mcp_path,
            "tools": list(mcp_tools),
        } if settings.mcp_enabled else {"enabled": False},
        "storm_cells": {
            "enabled": settings.storm_cells_enabled,
            "count": storm_cell_store.total_count if storm_cell_store is not None else 0,
            "last_updated": int(storm_cell_store.last_updated) if storm_cell_store is not None else 0,
            "per_region": await storm_cell_store.get_counts() if storm_cell_store is not None else {},
        } if settings.storm_cells_enabled else {"enabled": False},
    }


def _content_type(ext: str) -> str:
    return "image/webp" if ext == "webp" else "image/png"


_WEATHER_TILE_SIZES = (256, 512)
_WEATHER_TILE_FORMATS = ("png", "webp")
_WEATHER_TILE_MAX_AGE = 21_600
_WEATHER_ATTRIBUTION = "ECMWF IFS data via Open-Meteo"


def _weather_available_timestamps() -> list[int]:
    if ecmwf_grid is None:
        return []
    getter = getattr(ecmwf_grid, "available_timestamps", None)
    if getter is None:
        return []
    return sorted(int(timestamp) for timestamp in getter())


def _weather_model_version() -> str:
    if ecmwf_grid is None:
        return "unavailable"
    version = getattr(ecmwf_grid, "model_version", None)
    if version:
        return str(version)
    return f"{getattr(ecmwf_grid, 'reference_time', None)}:g{getattr(ecmwf_grid, 'grid_version', 0)}"


@router.get("/v2/weather/metadata.json", response_model=WeatherMetadataResponse)
async def weather_metadata(response: Response) -> WeatherMetadataResponse:
    """Metadata and legend definitions for scalar global weather tiles."""

    now = int(time.time())
    timestamps = _weather_available_timestamps()
    health = (
        ecmwf_grid.health_status(now)
        if ecmwf_grid is not None and hasattr(ecmwf_grid, "health_status")
        else {"stale": True}
    )
    default_timestamp = (
        ecmwf_grid.default_timestamp(now)
        if ecmwf_grid is not None and hasattr(ecmwf_grid, "default_timestamp")
        else (min(timestamps, key=lambda value: abs(value - now)) if timestamps else None)
    )
    fields = []
    for public_id, field in PUBLIC_WEATHER_FIELDS.items():
        if nwp_chain is not None and not nwp_chain.has_field(field):
            continue
        spec = field_spec(field)
        fields.append(
            WeatherFieldInfo(
                id=public_id,
                display_name=spec.public_name,
                unit=spec.unit,
                palette_ids=[palette.id for palette in palettes_for_field(field)],
            )
        )
    advertised_palette_ids = {
        palette_id for item in fields for palette_id in item.palette_ids
    }
    palettes = [
        WeatherPaletteInfo(
            id=palette.id,
            display_name=palette.display_name,
            unit=palette.unit,
            minimum=palette.minimum,
            maximum=palette.maximum,
            below_color=palette.below_color,
            above_color=palette.above_color,
            nodata_color=palette.nodata_color,
            opacity=palette.opacity,
            stops=[
                WeatherPaletteStop(value=stop.value, color=stop.color)
                for stop in palette.stops
            ],
        )
        for palette in WEATHER_PALETTES.values()
        if palette.id in advertised_palette_ids
    ]
    host = settings.public_url.rstrip("/")
    response.headers["Cache-Control"] = "public, max-age=60"
    return WeatherMetadataResponse(
        active_model_run=(
            ecmwf_grid.reference_time if ecmwf_grid is not None else None
        ),
        model_version=_weather_model_version(),
        generated=now,
        stale=bool(health.get("stale", True)),
        attribution=_WEATHER_ATTRIBUTION,
        fields=fields,
        available_timestamps=timestamps,
        default_timestamp=default_timestamp,
        palette_ids=[
            palette_id
            for palette_id in WEATHER_PALETTES
            if palette_id in advertised_palette_ids
        ],
        palettes=palettes,
        tile_url_template=(
            f"{host}/v2/weather/{{field}}/{{timestamp}}/{{size}}/"
            "{z}/{x}/{y}/{palette}.{ext}"
        ),
        point_url_template=(
            f"{host}/v2/weather/{{field}}/{{timestamp}}/point.json"
            "?lat={lat}&lon={lon}"
        ),
        sizes=list(_WEATHER_TILE_SIZES),
        formats=list(_WEATHER_TILE_FORMATS),
        min_zoom=0,
        max_zoom=settings.max_zoom,
    )


@router.get(
    "/v2/weather/{field}/{timestamp}/point.json",
    response_model=WeatherPointResponse,
)
async def weather_field_point(
    response: Response,
    field: str,
    timestamp: int,
    lat: float = Query(ge=-90.0, le=90.0),
    lon: float = Query(ge=-180.0, le=180.0),
) -> WeatherPointResponse:
    """Return the bilinearly interpolated physical value at one coordinate."""

    weather_field = PUBLIC_WEATHER_FIELDS.get(field)
    if weather_field is None:
        raise HTTPException(status_code=404, detail=f"Unknown weather field: {field}")
    timestamps = _weather_available_timestamps()
    if not timestamps or ecmwf_grid is None or nwp_chain is None:
        raise HTTPException(status_code=503, detail="Weather field data not available")
    if timestamp < timestamps[0] or timestamp > timestamps[-1]:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Timestamp {timestamp} outside available range "
                f"[{timestamps[0]}, {timestamps[-1]}]"
            ),
        )
    if not nwp_chain.has_field(weather_field):
        raise HTTPException(status_code=503, detail=f"Field {field} not available")

    try:
        value = await asyncio.to_thread(
            sample_scalar_weather_point,
            source=nwp_chain,
            field=weather_field,
            timestamp=timestamp,
            latitude=lat,
            longitude=lon,
        )
    except Exception as exc:
        logger.exception("Weather field point sampling failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Weather field point sampling failed",
        ) from exc
    now = int(time.time())
    health = (
        ecmwf_grid.health_status(now)
        if hasattr(ecmwf_grid, "health_status")
        else {"stale": True}
    )
    response.headers["Cache-Control"] = "no-store"
    return WeatherPointResponse(
        field=field,
        timestamp=timestamp,
        latitude=lat,
        longitude=lon,
        value=value,
        unit=field_spec(weather_field).unit,
        active_model_run=getattr(ecmwf_grid, "reference_time", None),
        stale=bool(health.get("stale", True)),
    )


@router.get(
    "/v2/weather/{field}/{timestamp}/{size}/{z}/{x}/{y}/{palette}.{ext}"
)
async def weather_field_tile(
    request: Request,
    field: str,
    timestamp: int,
    size: int,
    z: int = Path(ge=0),
    x: int = Path(ge=0),
    y: int = Path(ge=0),
    palette: str = Path(min_length=1),
    ext: str = Path(pattern=r"^(png|webp)$"),
    vectors: str = Query(default=""),
) -> Response:
    """Render a continuous global weather field without changing radar APIs."""

    weather_field = PUBLIC_WEATHER_FIELDS.get(field)
    if weather_field is None:
        raise HTTPException(status_code=404, detail=f"Unknown weather field: {field}")
    vector_style = ""
    if vectors in ("1", "true", "light"):
        vector_style = "light"
    elif vectors == "dark":
        vector_style = "dark"
    elif vectors:
        raise HTTPException(status_code=400, detail="Unsupported wind vector style")
    if vector_style and weather_field is not WeatherField.WIND_SPEED_10M:
        raise HTTPException(
            status_code=400,
            detail="Wind vectors are only supported for wind_speed_10m",
        )
    if size not in _WEATHER_TILE_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported tile size: {size}; use 256 or 512",
        )
    if z > settings.max_zoom:
        raise HTTPException(
            status_code=400,
            detail=f"Zoom {z} exceeds max {settings.max_zoom}",
        )
    max_tiles = 2**z
    if x >= max_tiles or y >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    selected_palette = WEATHER_PALETTES.get(palette)
    if selected_palette is None or selected_palette.field is not weather_field:
        allowed = [item.id for item in palettes_for_field(weather_field)]
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported palette for {field}; allowed: {', '.join(allowed)}",
        )
    timestamps = _weather_available_timestamps()
    if not timestamps or ecmwf_grid is None or nwp_chain is None:
        raise HTTPException(status_code=503, detail="Weather field data not available")
    if timestamp < timestamps[0] or timestamp > timestamps[-1]:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Timestamp {timestamp} outside available range "
                f"[{timestamps[0]}, {timestamps[-1]}]"
            ),
        )
    if not nwp_chain.has_field(weather_field):
        raise HTTPException(status_code=503, detail=f"Field {field} not available")

    model_version = getattr(ecmwf_grid, "model_version", None)
    if model_version is None:
        model_version = (
            f"{getattr(ecmwf_grid, 'reference_time', None)}:"
            f"g{getattr(ecmwf_grid, 'grid_version', 0)}"
        )
    cache_key = (
        "weather",
        field,
        model_version,
        timestamp,
        z,
        x,
        y,
        size,
        palette,
        vector_style,
        ext,
        (
            settings.webp_quality
            if ext == "webp"
            else (
                settings.weather_png_mode,
                settings.weather_png_colors,
                settings.weather_png_dither,
            )
        ),
        WEATHER_RENDERER_VERSION,
    )
    cached = tile_cache.get(cache_key) if tile_cache is not None else None
    if isinstance(cached, CachedRender):
        tile_bytes = cached.data
        etag = cached.etag
    else:
        async def _render_once() -> CachedRender:
            tile_bytes = await asyncio.to_thread(
                render_scalar_weather_tile,
                source=nwp_chain,
                field=weather_field,
                palette=selected_palette,
                timestamp=timestamp,
                z=z,
                x=x,
                y=y,
                tile_size=size,
                fmt=ext,
                vector_style=vector_style,
            )
            rendered = CachedRender(tile_bytes, compute_etag(tile_bytes))
            if tile_cache is not None:
                tile_cache.put(cache_key, rendered)
            return rendered

        try:
            rendered = await _weather_tile_singleflight(
                cache_key,
                _render_once,
            )
        except Exception as exc:
            logger.exception("Weather field tile render failed: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Weather field tile rendering failed",
            ) from exc
        tile_bytes = rendered.data
        etag = rendered.etag

    return conditional_response(
        request=request,
        body=tile_bytes,
        etag=etag,
        content_type=_content_type(ext),
        max_age=_WEATHER_TILE_MAX_AGE,
    )


@router.get("/public/weather-maps.json")
async def weather_maps() -> WeatherMapsResponse:
    """Rain Viewer-compatible metadata endpoint."""
    timestamps = await frame_store.get_timestamps()
    frame_versions = await frame_store.get_frame_versions()
    host = settings.public_url.rstrip("/")

    past = [
        RadarTimestamp(
            time=ts,
            path=f"/v2/radar/{ts}",
            version=f"r{ts}.{frame_versions.get(ts, 0)}",
        )
        for ts in sorted(timestamps)
    ]

    nowcast = []
    animation = None
    latest_observation = max(timestamps) if timestamps else 0
    nowcast_version = f"n{latest_observation}.{_weather_model_version()}"
    if nowcast_store is not None:
        nc_timestamps = await nowcast_store.get_timestamps()
        # Drop nowcast slots that duplicate the newest past frame: during
        # the nowcast-regeneration window the store is still anchored to
        # the previous cycle, so its first slot repeats the latest past
        # timestamp.
        latest_past = max(timestamps) if timestamps else None
        nowcast = [
            RadarTimestamp(
                time=ts,
                path=f"/v2/radar/{ts}",
                version=nowcast_version,
            )
            for ts in nc_timestamps
            if latest_past is None or ts > latest_past
        ]
        animation_frames = await nowcast_store.get_animation_frames()
        if animation_frames:
            animation = RadarAnimationData(
                substeps=settings.radar_animation_substeps,
                past=[
                    RadarTimestamp(
                        time=frame.timestamp,
                        path=f"/v2/radar/{frame.timestamp}",
                        version=f"a{frame.timestamp}",
                    )
                    for frame in animation_frames
                    if frame.period == "past"
                ],
                nowcast=[
                    RadarTimestamp(
                        time=frame.timestamp,
                        path=f"/v2/radar/{frame.timestamp}",
                        version=nowcast_version,
                    )
                    for frame in animation_frames
                    if frame.period == "forecast"
                ],
            )

    infrared = []
    # Catalog timestamps come from GMGSI LW since LW is the always-on
    # 24/7 baseline (VIS only carries the daytime half of the day).
    # When the satellite layer is disabled or unloaded the array is
    # empty and the tile endpoint returns 503.
    gmgsi_lw = satellite_grids.get("gmgsi_lw_grid") if satellite_grids else None
    if gmgsi_lw is not None and gmgsi_lw.timestamps:
        infrared = [
            RadarTimestamp(
                time=ts,
                path=f"/v2/satellite/{ts}",
                version=f"s{ts}",
            )
            for ts in gmgsi_lw.timestamps
        ]

    color_schemes = [
        ColorScheme(id=sid, name=name)
        for sid, name in SCHEME_NAMES.items()
    ]

    return WeatherMapsResponse(
        version="2.0",
        generated=int(time.time()),
        host=host,
        radar=RadarData(
            past=past,
            nowcast=nowcast,
            animation=animation,
            motion=RadarMotionData(
                path_template=(
                    "/v2/radar/motion/{from}/{to}/{size}/{z}/{x}/{y}.png"
                ),
                encoding=MOTION_ENCODING,
                vector_scale=MOTION_VECTOR_SCALE,
                vector_offset=MOTION_VECTOR_OFFSET,
                max_interval_seconds=settings.fetch_interval,
            ),
            colorSchemes=color_schemes,
        ),
        satellite=SatelliteData(infrared=infrared),
    )


@router.get(
    "/v2/radar/point-nowcast.json",
    response_model=RadarPointNowcastResponse,
)
async def radar_point_nowcast(
    response: Response,
    lat: float = Query(ge=-90.0, le=90.0),
    lon: float = Query(ge=-180.0, le=180.0),
    radius_km: float = Query(default=2.0, gt=0.0, le=10.0),
    past_minutes: int = Query(default=30, ge=0, le=120),
    future_minutes: int = Query(default=60, ge=0, le=60),
) -> RadarPointNowcastResponse:
    """Return observed and forecast radar summaries around one coordinate.

    Offsets are relative to the latest observed radar frame rather than wall
    clock time.  This keeps a delayed ingest visible through
    ``latest_age_seconds``/``stale`` instead of silently shortening or
    lengthening the advertised forecast horizon.
    """

    try:
        payload = await build_point_nowcast(
            frame_store=frame_store,
            nowcast_store=nowcast_store,
            enabled_regions=enabled_regions or [],
            lat=lat,
            lon=lon,
            radius_km=radius_km,
            past_minutes=past_minutes,
            future_minutes=future_minutes,
            noise_floor_dbz=settings.noise_floor_dbz,
            fetch_interval=settings.fetch_interval,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=503,
            detail="Radar observations not available",
        ) from exc
    # Exact coordinates may represent a user's location.  Keep the short cache
    # browser-private so shared intermediaries do not retain query URLs.
    response.headers["Cache-Control"] = "private, max-age=60"
    return RadarPointNowcastResponse(**payload)


async def _latest_timestamps_cached() -> list[int]:
    """Latest radar timestamps with a 5 s TTL.

    The tile hot path only needs the latest frame to pick the Cache-Control
    ``max_age`` bucket, so re-querying the store lock on every request is
    wasted contention.  Degrades to ``[]`` when the store isn't wired
    (``frame_store`` None -> no frames -> ``latest_ts`` None -> 300 s
    bucket), matching the single-mode no-store behaviour.
    """
    global _latest_ts_cache
    now = time.monotonic()
    if _latest_ts_cache is not None and now - _latest_ts_cache[0] < _LATEST_TS_TTL:
        return _latest_ts_cache[1]
    if frame_store is None:
        _latest_ts_cache = (now, [])
        return []
    timestamps = await frame_store.get_timestamps()
    _latest_ts_cache = (now, timestamps)
    return timestamps


async def _resolve_radar_timestamp(timestamp: int) -> int:
    """Resolve the ``0`` = "latest past frame" alias to a real timestamp.

    The coverage endpoint already hardcodes ``0`` in the timestamp slot
    to mean "latest" (RainViewer's own convention); this extends the same
    convention to the radar tile/window routes.  Resolution happens at
    route entry so every downstream consumer (cache keys, shared-store
    keys, the max-age bucket, NWP/snow sampling) sees the resolved
    timestamp.  Raises 404 when the store holds no frames, matching the
    endpoint's unknown-timestamp behaviour.
    """
    if timestamp != 0:
        return timestamp
    if frame_store is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    latest = await frame_store.get_latest_frame()
    if latest is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    return latest.timestamp


def _present_and_hash(geom, **kwargs) -> tuple[bytes, str]:
    """Run ``present_tile`` and hash the result into an ETag.

    Kept together so both run off the event loop: the SHA-256 of a tile
    is a per-request cost that would otherwise stall every miss on the
    loop.
    """
    tile_bytes = present_tile(geom, **kwargs)
    return tile_bytes, compute_etag(tile_bytes)


def _shared_tile_key(
    timestamp, version, z, x, y, tile_size, smooth, snow, color, ext,
    display_min_dbz=None,
) -> str:
    """Shared-store key for an encoded radar tile.

    Folds every input that determines the encoded bytes (including the
    frame's content version) so a merge/eviction or config change re-keys
    the tile instead of serving stale bytes.
    """
    return (
        f"{timestamp}-v{version}-{z}-{x}-{y}-{tile_size}-"
        f"{int(smooth)}{int(snow)}-{color}-{ext}-q{settings.webp_quality}"
        f"-m{display_min_dbz}"
    )


def _shared_overlay_key(
    timestamp, frame_version, flow_version, cells_version, z, x, y,
    tile_size, smooth, snow, color, ext, arrow_style, cell_style,
    display_min_dbz=None,
) -> str:
    """Shared-store key for an encoded overlay tile.

    Same content-versioning principle as ``_shared_tile_key`` extended
    with flow/cells versions so a pipeline regeneration re-keys overlay
    tiles instead of serving stale bytes; the leading timestamp keeps
    ``_ts_of`` sharding and ``invalidate_timestamp`` prefix sweeps
    working.
    """
    return (
        f"{timestamp}-v{frame_version}-f{flow_version}-c{cells_version}-"
        f"{z}-{x}-{y}-{tile_size}-{int(smooth)}{int(snow)}-{color}-{ext}-"
        f"q{settings.webp_quality}-a{arrow_style}-k{cell_style}"
        f"-m{display_min_dbz}"
    )


_INT_RE = re.compile(r"^-?\d+$")


def _parse_tile_or_window_coords(x: str, y: str) -> tuple[str, int, int, float, float]:
    """RainViewer-style tile/window coordinate discrimination.

    Both strings containing a dot -> ("window", 0, 0, lat, lon) with
    float(x)/float(y) as (lat, lon); both must be finite and the latitude
    must lie within [-90.0, 90.0] (the longitude may be any finite value,
    it is normalized downstream).  Both matching a plain integer
    (``_INT_RE``) -> ("tile", int(x), int(y), 0.0, 0.0); either negative
    is rejected, restoring the lower bound the old ``Path(ge=0)`` typing
    enforced.  Anything else (mixed dot/int, unparseable, non-finite) is
    invalid.

    NOTE: RainViewer parity - a coordinate without a dot is an integer
    tile index even if it names a latitude ("55" is tile x=55, not lat
    55).
    """
    if "." in x and "." in y:
        try:
            lat = float(x)
            lon = float(y)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid tile coordinates")
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise HTTPException(status_code=400, detail="Invalid tile coordinates")
        if not -90.0 <= lat <= 90.0:
            raise HTTPException(status_code=400, detail="Latitude out of range")
        return "window", 0, 0, lat, lon
    if _INT_RE.match(x) and _INT_RE.match(y):
        xi = int(x)
        yi = int(y)
        if xi < 0 or yi < 0:
            raise HTTPException(
                status_code=400, detail="Tile coordinates out of range",
            )
        return "tile", xi, yi, 0.0, 0.0
    raise HTTPException(status_code=400, detail="Invalid tile coordinates")


def _shared_get_and_hash(store, key: str):
    """Read a shared tile and hash it off the event loop; None on miss."""
    data = store.get(key)
    if data is None:
        return None
    return data, compute_etag(data)


async def _present_tile_async(geom, **kwargs) -> tuple[bytes, str]:
    """Run ``present_tile`` + ETag hash off the event loop.

    Multi-mode render workers get a dedicated present pool
    (``routes.present_executor``) so cheap colorize/encode jobs never queue
    behind long geometry computes on the shared default executor.  Single
    mode leaves ``present_executor`` None and falls back to
    ``asyncio.to_thread`` - byte-identical to the pre-split behaviour.
    """
    submitted_ns = time.perf_counter_ns()

    def _run_present() -> tuple[bytes, str]:
        started_ns = time.perf_counter_ns()
        stage_timings = kwargs.get("stage_timings")
        if stage_timings is not None:
            stage_timings["present_queue"] = max(0, started_ns - submitted_ns)
        return _present_and_hash(geom, **kwargs)

    if present_executor is not None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(present_executor, _run_present)
    return await asyncio.to_thread(_run_present)


async def _radar_window(
    request: Request,
    timestamp: int,
    size: int,
    z: int,
    lat: float,
    lon: float,
    color: int,
    smooth_snow: str,
    ext: str,
) -> Response:
    """Lat/lon-centered window mode for the radar tile route.

    Triggered when both ``/v2/radar`` ``x``/``y`` path segments contain a
    dot: the request names a (lat, lon) point and the response is a
    ``tile_size`` x ``tile_size`` canvas centered on that point (see
    ``coordinates.window_origin`` / ``window.compute_window_geometry``),
    stitched from ordinary integer-tile geometry at the same zoom.

    Deliberate differences from tile mode: overlay query params are
    silently ignored, and the shared tile store, tile warmer, and the
    request tracker's per-tile counters are never touched.
    """
    t0 = time.perf_counter_ns()

    parts = smooth_snow.split("_")
    smooth = parts[0] == "1"
    snow = parts[1] == "1" if len(parts) > 1 else False

    tile_size = 512 if size >= 512 else 256
    px0, py0 = window_origin(lat, lon, z, tile_size)

    present_key = (
        timestamp, "win", z, px0, py0, tile_size, smooth, snow,
        color, ext, settings.webp_quality,
    )
    cached = tile_cache.get(present_key)
    if isinstance(cached, CachedRender):
        tile_bytes = cached.data
        etag = cached.etag
    else:
        # Frame resolution identical to tile mode: radar store first,
        # nowcast store fallback, 404 when neither has the timestamp.
        frame = await frame_store.get_frame(timestamp)
        nowcast_blend = None
        if frame is None and nowcast_store is not None:
            nc_frame, nowcast_blend = await nowcast_store.get_frame(timestamp)
            if nc_frame is not None:
                frame = nc_frame
        if frame is None and nowcast_store is not None:
            animation_frame = await nowcast_store.get_animation_frame(timestamp)
            if animation_frame is not None:
                frame = animation_frame
                if animation_frame.period == "forecast":
                    nowcast_blend = animation_frame.blend_weight
        if frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")

        async def get_component(tx: int, ty: int) -> TileGeometry:
            geom_key = (timestamp, z, tx, ty, tile_size, smooth, snow)
            geom = tile_cache.get(geom_key)
            if geom is not None:
                return geom
            label = transparent_fast_path_label(
                frame.regions, z, tx, ty, enabled_regions, nwp_chain,
                precip_mask, timestamp, nowcast_blend,
            )
            if label is not None:
                geom = TileGeometry.transparent(tile_size, fast_path=label)
            else:
                geom = await asyncio.to_thread(
                    compute_tile_geometry,
                    frame_regions=frame.regions,
                    z=z, x=tx, y=ty,
                    tile_size=tile_size,
                    smooth=smooth,
                    snow=snow,
                    nwp_chain=nwp_chain,
                    enabled_regions=enabled_regions,
                    frame_timestamp=timestamp,
                    nowcast_blend=nowcast_blend,
                    precip_mask=precip_mask,
                )
            tile_cache.put(geom_key, geom)
            return geom

        geom = await window.compute_window_geometry(
            get_component, z, px0, py0, tile_size,
        )

        if geom.is_transparent:
            # Same inline constant serve as tile mode: the transparent
            # window's encoded bytes are a process constant per
            # (tile_size, ext); the present entry is primed so repeats
            # hit the present cache.
            memo_key = (tile_size, ext)
            memoized = _TRANSPARENT_RENDER_MEMO.get(memo_key)
            if memoized is None:
                tile_bytes = _transparent_tile(tile_size, ext)
                etag = compute_etag(tile_bytes)
                _TRANSPARENT_RENDER_MEMO[memo_key] = (tile_bytes, etag)
            else:
                tile_bytes, etag = memoized
            tile_cache.put(present_key, CachedRender(data=tile_bytes, etag=etag))
        else:
            tile_bytes, etag = await _present_tile_async(
                geom,
                color_scheme=color,
                fmt=ext,
                arrow_style="",
            )
            tile_cache.put(present_key, CachedRender(data=tile_bytes, etag=etag))

    # Historical frames are immutable once backfill is complete - cache
    # them for their full 2-hour lifetime.  Latest and nowcast frames
    # still evolve (mirror of the tile-mode tail).
    timestamps = await _latest_timestamps_cached()
    latest_ts = max(timestamps) if timestamps else None
    max_age = 7200 if (latest_ts is not None and timestamp < latest_ts) else 300

    # Request latency: request-total-only form (no compute/present stages
    # are tracked separately in window mode).
    if tile_request_tracker is not None:
        tile_request_tracker.record_latency(
            time.perf_counter_ns() - t0, None, None,
        )

    return conditional_response(
        request=request,
        body=tile_bytes,
        etag=etag,
        content_type=_content_type(ext),
        max_age=max_age,
        extra_headers={"X-Frame-Timestamp": str(timestamp)},
    )


async def _motion_geometry(
    timestamp: int,
    *,
    z: int,
    x: int,
    y: int,
    tile_size: int,
):
    """Load one cached post-composite geometry for the motion endpoint."""
    # Match the public site's smooth + snow-capable radar geometry so the
    # expensive composite can be reused. Its overlap padding also gives
    # optical flow context across tile edges, avoiding visible motion seams.
    geom_key = (timestamp, z, x, y, tile_size, True, True)
    geometry = tile_cache.get(geom_key)
    if geometry is not None:
        return geometry

    async def _compute_once():
        frame = await frame_store.get_frame(timestamp)
        nowcast_blend = None
        if frame is None and nowcast_store is not None:
            frame, nowcast_blend = await nowcast_store.get_frame(timestamp)
        if frame is None and nowcast_store is not None:
            animation_frame = await nowcast_store.get_animation_frame(timestamp)
            if animation_frame is not None:
                frame = animation_frame
                if animation_frame.period == "forecast":
                    nowcast_blend = animation_frame.blend_weight
        if frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")

        computed = await asyncio.to_thread(
            compute_tile_geometry,
            frame_regions=frame.regions,
            z=z,
            x=x,
            y=y,
            tile_size=tile_size,
            smooth=True,
            snow=True,
            nwp_chain=nwp_chain,
            enabled_regions=enabled_regions,
            frame_timestamp=timestamp,
            nowcast_blend=nowcast_blend,
            precip_mask=precip_mask,
        )
        tile_cache.put(geom_key, computed)
        return computed

    geometry, _leader = await _singleflight(
        _geometry_flights, geom_key, _compute_once,
    )
    return geometry


@router.get("/v2/radar/motion/{previous}/{following}/{size}/{z}/{x}/{y}.png")
async def radar_motion_tile(
    request: Request,
    previous: int = Path(ge=1_000_000_000, le=9_999_999_999),
    following: int = Path(ge=1_000_000_000, le=9_999_999_999),
    size: int = Path(ge=256, le=512),
    z: int = Path(ge=0),
    x: int = Path(ge=0),
    y: int = Path(ge=0),
) -> Response:
    """Return signed pixel displacement between two adjacent radar frames."""
    if following <= previous:
        raise HTTPException(status_code=400, detail="Motion timestamps out of order")
    if following - previous > settings.fetch_interval:
        raise HTTPException(status_code=400, detail="Motion interval is too large")
    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")
    max_tiles = 2**z
    if x >= max_tiles or y >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    tile_size = 512 if size >= 512 else 256
    cache_key = (
        "radar-motion",
        MOTION_RENDERER_VERSION,
        previous,
        following,
        z,
        x,
        y,
        tile_size,
    )
    cached = tile_cache.get(cache_key)
    if isinstance(cached, CachedRender):
        rendered = cached
    else:
        previous_geometry, following_geometry = await asyncio.gather(
            _motion_geometry(previous, z=z, x=x, y=y, tile_size=tile_size),
            _motion_geometry(following, z=z, x=x, y=y, tile_size=tile_size),
        )
        tile_bytes = await asyncio.to_thread(
            render_motion_tile,
            previous_geometry,
            following_geometry,
        )
        rendered = CachedRender(data=tile_bytes, etag=compute_etag(tile_bytes))
        tile_cache.put(cache_key, rendered)

    timestamps = await _latest_timestamps_cached()
    latest_ts = max(timestamps) if timestamps else None
    max_age = 7200 if latest_ts is not None and following < latest_ts else 300
    return conditional_response(
        request=request,
        body=rendered.data,
        etag=rendered.etag,
        content_type="image/png",
        max_age=max_age,
    )


@router.get("/v2/radar/{timestamp}/{size}/{z}/{x}/{y}/{color}/{smooth_snow}.{ext}")
async def radar_tile(
    request: Request,
    timestamp: int,
    size: int = Path(ge=256, le=512),
    z: int = Path(ge=0),
    x: str = Path(...),
    y: str = Path(...),
    color: int = Path(ge=0, le=255),
    smooth_snow: str = Path(pattern=r"^\d+_\d+$"),
    ext: str = Path(pattern=r"^(png|webp)$"),
    arrows: str = Query(default=""),
    cells: str = Query(default=""),
    min_dbz: float | None = Query(default=None, ge=-32.0, le=95.5),
) -> Response:
    """Rain Viewer-compatible tile endpoint."""
    t0 = time.perf_counter_ns()
    logger.debug("Tile request: z=%d x=%s y=%s color=%d smooth_snow=%s ext=%s", z, x, y, color, smooth_snow, ext)
    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")

    timestamp = await _resolve_radar_timestamp(timestamp)

    mode, xi, yi, lat, lon = _parse_tile_or_window_coords(x, y)
    if mode == "window":
        return await _radar_window(
            request, timestamp, size, z, lat, lon, color, smooth_snow, ext,
        )

    max_tiles = 2**z
    if xi >= max_tiles or yi >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    parts = smooth_snow.split("_")
    smooth = parts[0] == "1"
    snow = parts[1] == "1" if len(parts) > 1 else False

    tile_size = 512 if size >= 512 else 256

    arrow_style = ""
    if arrows in ("1", "true", "light"):
        arrow_style = "light"
    elif arrows == "dark":
        arrow_style = "dark"

    cell_style = ""
    if cells in ("1", "true", "light"):
        cell_style = "light"
    elif cells == "dark":
        cell_style = "dark"

    display_min_dbz = (
        max(settings.noise_floor_dbz, min_dbz)
        if min_dbz is not None else None
    )

    # Plain tile: no overlays requested.  Computed here (before the frame
    # fetch) because the shared-store lookup only serves plain tiles and
    # needs the flag for its guard.
    is_plain = not arrows and not cells

    # Geometry cache: keyed only on inputs that affect the sampled values
    # (radar source + viewport + smoothing + snow-mask presence).  Color
    # scheme, output format, and arrow style apply per-request in
    # ``present_tile`` so a single cached entry serves every visual
    # variant of the same viewport.
    geom_key = (timestamp, z, xi, yi, tile_size, smooth, snow)
    geom = tile_cache.get(geom_key)

    # Present-stage cache key: one entry per visual variant of the same
    # geometry.  Stores the encoded bytes plus the ETag so a present
    # cache hit skips both ``present_tile`` and the ETag hash.  Hoisted
    # here (before the shared-store lookup) so the early-serve fast path
    # below can use it without recomputing the tuple.
    present_key = (
        timestamp, z, xi, yi, tile_size, smooth, snow,
        color, ext, settings.webp_quality, display_min_dbz,
    )

    # Geometry-stage cache outcome: this is the meaningful hit/miss for
    # "is the fast path helping".  Batched with the (z, x, y) counter so
    # one lock acquisition covers both.
    if tile_request_tracker is not None:
        tile_request_tracker.record_request(z, xi, yi, cache_hit=geom is not None)

    # Early-serve fast path: a plain request whose geometry is already in
    # memory and whose encoded bytes are present-cached skips the shared-
    # store lookup, frame fetch, geometry compute, and present entirely.
    # The ``geom is not None`` gate is required: it keeps the nowcast-
    # timestamp edge case on the existing path (with a geom hit,
    # ``need_frame`` is False and the warmer hook below resolves
    # ``frame_type`` via ``_latest_timestamps_cached``).
    present_cache_hit = False
    if is_plain and geom is not None:
        cached = tile_cache.get(present_key)
        if isinstance(cached, CachedRender):
            tile_bytes = cached.data
            etag = cached.etag
            present_cache_hit = True

    # ``need_frame``/``is_nowcast`` live above the branch because the
    # warmer hook below runs on every path; on a shared hit (past frames
    # only) ``is_nowcast`` stays False, which is exactly what a plain
    # cached-hit request resolves to.
    is_nowcast = False
    need_frame = geom is None or bool(arrow_style) or bool(cell_style)
    compute_ns = None
    present_ns = None
    stage_timings: dict[str, int] = {}
    if not present_cache_hit:
        # Shared-store lookup: plain past-frame tiles only.  A hit here still
        # counts as a geometry miss in ``record_request`` above (accepted -
        # it avoided the compute, not the lookup).  The key folds the frame's
        # content version so a merge/eviction re-keys the tile.
        shared_hit = None
        if shared_tile_store is not None and is_plain and frame_store is not None:
            version = frame_store.frame_version(timestamp)
            if version is not None:  # past frames only; nowcast ts has no version
                shared_key = _shared_tile_key(
                    timestamp, version, z, xi, yi, tile_size, smooth, snow,
                    color, ext, display_min_dbz,
                )
                if io_executor is not None:
                    loop = asyncio.get_running_loop()
                    shared_hit = await loop.run_in_executor(
                        io_executor, _shared_get_and_hash, shared_tile_store, shared_key,
                    )
                else:
                    shared_hit = await asyncio.to_thread(_shared_get_and_hash, shared_tile_store, shared_key)

        if shared_hit is not None:
            # Shared hit: the published bytes (and ETag) are byte-identical to
            # a fresh render, so skip frame fetch, geometry compute, overlays,
            # and present entirely.  Prime the in-memory present cache so
            # same-worker repeats hit RAM instead of the shared volume.  The
            # warmer hook below stays reachable from every path; in practice it
            # never fires here because the shared store is only wired in multi
            # mode, where ``tile_warmer`` is None.
            tile_bytes, etag = shared_hit
            tile_cache.put(present_key, CachedRender(data=tile_bytes, etag=etag))
        else:
            # We need the radar frame whenever geometry must be computed AND
            # whenever an overlay is requested: arrows need live frame data +
            # flow fields, and cells need ``frame.regions`` to decide which
            # regions actually carry data on this tile (without it
            # ``_draw_storm_cells`` sees an empty region list and draws
            # nothing).  Skip the fetch on pure cache hits without overlays -
            # that's the hot path Merry Sky-style clients exercise.
            frame = None
            nowcast_blend = None
            if need_frame:
                frame = await frame_store.get_frame(timestamp)
                if frame is None and nowcast_store is not None:
                    nc_frame, nowcast_blend = await nowcast_store.get_frame(timestamp)
                    if nc_frame is not None:
                        frame = nc_frame
                        is_nowcast = True
                if frame is None and nowcast_store is not None:
                    animation_frame = await nowcast_store.get_animation_frame(timestamp)
                    if animation_frame is not None:
                        frame = animation_frame
                        is_nowcast = animation_frame.period == "forecast"
                        nowcast_blend = (
                            animation_frame.blend_weight if is_nowcast else None
                        )
                if frame is None:
                    raise HTTPException(status_code=404, detail="Frame not found")

            if geom is None:
                # Transparent fast-path gate: microsecond-scale pure reads
                # (region-overlap math, dict membership, a memmap slice +
                # any()), safe on the event loop, and keeps the compute
                # pool clear of no-op ocean tiles.
                label = transparent_fast_path_label(
                    frame.regions, z, xi, yi, enabled_regions, nwp_chain,
                    precip_mask, timestamp, nowcast_blend,
                )
                if label is not None:
                    # Cold-compute-equivalent: the request paid for the
                    # empty-tile decision here (no pool compute ran, so
                    # compute_ns stays None).
                    geom = TileGeometry.transparent(tile_size, fast_path=label)
                    tile_cache.put(geom_key, geom)
                    if tile_request_tracker is not None and geom.fast_path is not None:
                        tile_request_tracker.record_fast_path(geom.fast_path)
                else:
                    async def _compute_once():
                        local_timings: dict[str, int] = {}
                        compute_start = time.perf_counter_ns()
                        def _run_compute():
                            started_ns = time.perf_counter_ns()
                            local_timings["compute_queue"] = max(
                                0, started_ns - compute_start,
                            )
                            return compute_tile_geometry(
                                frame_regions=frame.regions,
                                z=z, x=xi, y=yi,
                                tile_size=tile_size,
                                smooth=smooth,
                                snow=snow,
                                nwp_chain=nwp_chain,
                                enabled_regions=enabled_regions,
                                frame_timestamp=timestamp,
                                nowcast_blend=nowcast_blend,
                                precip_mask=precip_mask,
                                stage_timings=local_timings,
                            )

                        if render_queue is not None:
                            async with render_queue:
                                computed = await asyncio.to_thread(_run_compute)
                        else:
                            computed = await asyncio.to_thread(_run_compute)
                        elapsed_ns = time.perf_counter_ns() - compute_start
                        tile_cache.put(geom_key, computed)
                        return computed, elapsed_ns, local_timings

                    result, flight_leader = await _singleflight(
                        _geometry_flights, geom_key, _compute_once,
                    )
                    geom, flight_compute_ns, flight_timings = result
                    if flight_leader:
                        compute_ns = flight_compute_ns
                        stage_timings.update(flight_timings)
                    # Only fire on the cold-compute path: a fast-path label here means
                    # this request actually paid for the empty-tile work (cache hits
                    # of a previously-computed transparent geometry are already counted
                    # by ``record_request`` above, not a fast-path firing now).
                    if tile_request_tracker is not None and geom.fast_path is not None:
                        tile_request_tracker.record_fast_path(geom.fast_path)

            # Versions are read before the flow/cell fetches so a mid-flight
            # pipeline swap keys the rendered tile conservatively; lookups
            # always use the current version, so stale entries become
            # unreachable on the next request.
            flow_v = nowcast_store.flow_version if nowcast_store is not None else 0
            cells_v = storm_cell_store.cells_version if storm_cell_store is not None else 0

            flow_regions = None
            nwp_flow = None
            if arrow_style:
                if nowcast_store is not None:
                    flow_regions = await nowcast_store.get_flows() or None
                    nwp_flow = await nowcast_store.get_nwp_flow()

            cells_by_region = None
            cell_counts = None
            if cell_style and storm_cell_store is not None:
                # Only show cells on the frame the detection actually ran on --
                # showing current-detected cells on past or nowcast frames is
                # misleading (the cells represent "what storms are detected RIGHT
                # NOW", not historical positions).
                if timestamp == storm_cell_store.detected_at_timestamp:
                    cells_by_region = await storm_cell_store.get_cells() or None
                    cell_counts = await storm_cell_store.get_counts() or None

            # Effective overlay styles as actually passed to ``present_tile``:
            # an arrows/cells request degrades to plain when no flow or cell
            # data is available for this request.
            eff_arrow = arrow_style if (flow_regions or nwp_flow is not None) else ""
            eff_cells = cell_style if cells_by_region else ""

            if is_plain or not (eff_arrow or eff_cells):
                # An overlay request with no flow/cell data available also lands
                # here - it falls through to the exact plain present path (same
                # present_key, same cache entry) rather than creating a duplicate.
                if geom.fast_path is not None:
                    # Inline serve: a fully transparent tile's encoded bytes are
                    # a process constant per (tile_size, ext), so skip the
                    # present-pool thread hop and the per-request SHA-256 (the
                    # dominant global request class).  The entry is primed into
                    # the present cache so repeats hit the early-serve path
                    # above.  present_ns stays None - no present stage ran - and
                    # the shared-store publish is skipped too: a transparent
                    # constant costs nothing for any worker to regenerate
                    # locally, so publishing it would only churn the shared
                    # volume.
                    memo_key = (tile_size, ext)
                    memoized = _TRANSPARENT_RENDER_MEMO.get(memo_key)
                    if memoized is None:
                        tile_bytes = _transparent_tile(tile_size, ext)
                        etag = compute_etag(tile_bytes)
                        _TRANSPARENT_RENDER_MEMO[memo_key] = (tile_bytes, etag)
                    else:
                        tile_bytes, etag = memoized
                    tile_cache.put(present_key, CachedRender(data=tile_bytes, etag=etag))
                else:
                    cached = tile_cache.get(present_key)
                    if isinstance(cached, CachedRender):
                        tile_bytes = cached.data
                        etag = cached.etag
                    else:
                        present_start = time.perf_counter_ns()
                        tile_bytes, etag = await _present_tile_async(
                            geom,
                            color_scheme=color,
                            fmt=ext,
                            display_min_dbz=display_min_dbz,
                            arrow_style=eff_arrow,
                            flow_regions=flow_regions,
                            frame_regions=frame.regions if frame is not None else None,
                            enabled_regions=enabled_regions,
                            nwp_flow=nwp_flow,
                            nwp_chain=nwp_chain,
                            frame_timestamp=timestamp,
                            z=z, x=xi, y=yi,
                            cell_style=eff_cells,
                            cells_by_region=cells_by_region,
                            cell_counts=cell_counts,
                            stage_timings=stage_timings,
                        )
                        present_ns = time.perf_counter_ns() - present_start
                        tile_cache.put(present_key, CachedRender(data=tile_bytes, etag=etag))
                        # Publish the fresh encode to the shared store for the other
                        # workers, fire-and-forget so the response never waits on the
                        # shared-volume write; the set holds references so tasks
                        # can't be GC'd mid-flight.  Never fires for nowcast tiles
                        # (their timestamp has no frame version).  A version bump
                        # between lookup and publish writes a stale entry, but the
                        # render worker's poller detects the bump within one poll
                        # interval (~1 s) and ``invalidate_timestamp`` sweeps every
                        # key for that timestamp, so the stale window is bounded by
                        # the poll cadence (pruning covers orphaned entries after
                        # eviction).
                        if shared_tile_store is not None and frame_store is not None:
                            version = frame_store.frame_version(timestamp)
                            if version is not None:
                                key = _shared_tile_key(
                                    timestamp, version, z, xi, yi, tile_size,
                                    smooth, snow, color, ext, display_min_dbz,
                                )
                                if io_executor is not None:
                                    loop = asyncio.get_running_loop()
                                    task = asyncio.ensure_future(
                                        loop.run_in_executor(io_executor, shared_tile_store.publish, key, tile_bytes)
                                    )
                                else:
                                    task = asyncio.ensure_future(
                                        asyncio.to_thread(shared_tile_store.publish, key, tile_bytes)
                                    )
                                _pending_shared_publishes.add(task)
                                task.add_done_callback(_pending_shared_publishes.discard)
            else:
                # Overlay tiles are cached and shared under flow/cells-
                # versioned keys, so staleness is bounded by the state poll
                # interval (~1 s) plus publish lag - the same guarantee the
                # nowcast overlay cache already had.  The flow/cells
                # versions re-key the entry when the pipeline regenerates
                # flows/detections (the poller still sweeps nowcast
                # timestamps per cycle as before), and past-frame overlay
                # tiles are now cached too - previously re-rendered per
                # request.
                overlay_key = present_key + (eff_arrow, eff_cells, flow_v, cells_v)
                cached = tile_cache.get(overlay_key)
                if isinstance(cached, CachedRender):
                    tile_bytes = cached.data
                    etag = cached.etag
                else:
                    # Shared-store lookup: past-frame overlays only (a
                    # nowcast timestamp has no frame version and keeps the
                    # in-worker-only cache).  Same executor pattern as the
                    # plain path.
                    ov_hit = None
                    if shared_tile_store is not None and frame_store is not None:
                        version = frame_store.frame_version(timestamp)
                        if version is not None:
                            ov_key = _shared_overlay_key(
                                timestamp, version, flow_v, cells_v, z, xi, yi,
                                tile_size, smooth, snow, color, ext,
                                eff_arrow, eff_cells, display_min_dbz,
                            )
                            if io_executor is not None:
                                loop = asyncio.get_running_loop()
                                ov_hit = await loop.run_in_executor(
                                    io_executor, _shared_get_and_hash, shared_tile_store, ov_key,
                                )
                            else:
                                ov_hit = await asyncio.to_thread(_shared_get_and_hash, shared_tile_store, ov_key)
                    if ov_hit is not None:
                        tile_bytes, etag = ov_hit
                        tile_cache.put(overlay_key, CachedRender(data=tile_bytes, etag=etag))
                    else:
                        present_start = time.perf_counter_ns()
                        tile_bytes, etag = await _present_tile_async(
                            geom,
                            color_scheme=color,
                            fmt=ext,
                            display_min_dbz=display_min_dbz,
                            arrow_style=eff_arrow,
                            flow_regions=flow_regions,
                            frame_regions=frame.regions if frame is not None else None,
                            enabled_regions=enabled_regions,
                            nwp_flow=nwp_flow,
                            nwp_chain=nwp_chain,
                            frame_timestamp=timestamp,
                            z=z, x=xi, y=yi,
                            cell_style=eff_cells,
                            cells_by_region=cells_by_region,
                            cell_counts=cell_counts,
                            stage_timings=stage_timings,
                        )
                        present_ns = time.perf_counter_ns() - present_start
                        tile_cache.put(overlay_key, CachedRender(data=tile_bytes, etag=etag))
                        # Publish the fresh encode to the shared store for the
                        # other workers, fire-and-forget so the response never
                        # waits on the shared-volume write; the set holds
                        # references so tasks can't be GC'd mid-flight.  Past
                        # frames only (a nowcast timestamp has no frame
                        # version).  A version bump between lookup and publish
                        # writes a stale entry, but the flow/cells versions in
                        # the key make it unreachable as soon as the next
                        # request reads the current versions.
                        if shared_tile_store is not None and frame_store is not None:
                            version = frame_store.frame_version(timestamp)
                            if version is not None:
                                ov_key = _shared_overlay_key(
                                    timestamp, version, flow_v, cells_v, z, xi, yi,
                                    tile_size, smooth, snow, color, ext,
                                    eff_arrow, eff_cells, display_min_dbz,
                                )
                                if io_executor is not None:
                                    loop = asyncio.get_running_loop()
                                    task = asyncio.ensure_future(
                                        loop.run_in_executor(io_executor, shared_tile_store.publish, ov_key, tile_bytes)
                                    )
                                else:
                                    task = asyncio.ensure_future(
                                        asyncio.to_thread(shared_tile_store.publish, ov_key, tile_bytes)
                                    )
                                _pending_shared_publishes.add(task)
                                task.add_done_callback(_pending_shared_publishes.discard)

    if tile_warmer is not None:
        # When the cache hit short-circuited the frame fetch, we still
        # need a frame_type for the warmer.  Cheap lookup against the
        # in-memory timestamp list.
        if not need_frame:
            past_timestamps = await _latest_timestamps_cached()
            is_nowcast = timestamp not in past_timestamps
        asyncio.ensure_future(
            tile_warmer.warm(
                triggered_timestamp=timestamp,
                z=z, x=xi, y=yi,
                tile_size=tile_size,
                smooth=smooth,
                snow=snow,
                frame_type="nowcast" if is_nowcast else "past",
            )
        )

    # Historical frames are immutable once backfill is complete — cache them
    # for their full 2-hour lifetime.  Latest and nowcast frames still evolve.
    timestamps = await _latest_timestamps_cached()
    latest_ts = max(timestamps) if timestamps else None
    max_age = 7200 if (latest_ts is not None and timestamp < latest_ts) else 300

    # Request latency: the request is always counted; compute/present only
    # when that stage actually ran (None on cache hits).
    if tile_request_tracker is not None:
        tile_request_tracker.record_latency(
            time.perf_counter_ns() - t0, compute_ns, present_ns,
            stages_ns=stage_timings,
        )

    return conditional_response(
        request=request,
        body=tile_bytes,
        etag=etag,
        content_type=_content_type(ext),
        max_age=max_age,
        extra_headers={"X-Frame-Timestamp": str(timestamp)},
    )


async def _coverage_window(
    request: Request, size: int, z: int, lat: float, lon: float,
) -> Response:
    """Lat/lon-centered window variant of the coverage route.

    Stitches per-tile coverage RGBA arrays (``compute_coverage_rgba``)
    onto a ``tile_size`` x ``tile_size`` canvas centered on the given
    lat/lon point.  No component-level caching in v1 - the window is
    recomputed per unique origin and cached whole under a ``"cov"``
    namespaced key.
    """
    tile_size = 512 if size >= 512 else 256
    px0, py0 = window_origin(lat, lon, z, tile_size)

    frame = await frame_store.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No radar data available")

    cache_key = ("cov", frame.timestamp, "win", z, px0, py0, tile_size)
    cached = tile_cache.get(cache_key)
    if cached is not None:
        if isinstance(cached, CachedRender):
            tile_bytes = cached.data
            etag = cached.etag
        else:
            # Legacy raw-bytes entry (pre-ETag cache format).
            tile_bytes = cached
            etag = compute_etag(cached)
        return conditional_response(
            request=request,
            body=tile_bytes,
            etag=etag,
            content_type="image/png",
            max_age=300,
        )

    keys: list[tuple[int, int]] = list(dict.fromkeys(
        (s.tx, s.ty)
        for s in window.covered_tiles(z, px0, py0, tile_size, tile_size)
    ))
    rgbas = await asyncio.gather(*[
        asyncio.to_thread(
            compute_coverage_rgba,
            frame_regions=frame.regions,
            z=z, x=tx, y=ty,
            tile_size=tile_size,
            enabled_regions=enabled_regions,
        )
        for tx, ty in keys
    ])
    canvas = window.stitch_coverage(dict(zip(keys, rgbas)), z, px0, py0, tile_size)

    if not canvas.any():
        tile_bytes = _transparent_tile(tile_size, "png")
    else:
        img = Image.fromarray(canvas, "RGBA")
        tile_bytes = _encode_image(img, "png")
    etag = compute_etag(tile_bytes)
    tile_cache.put(cache_key, CachedRender(data=tile_bytes, etag=etag))

    return conditional_response(
        request=request,
        body=tile_bytes,
        etag=etag,
        content_type="image/png",
        max_age=300,
    )


@router.get("/v2/coverage/0/{size}/{z}/{x}/{y}/0/0_0.png")
async def coverage_tile(
    request: Request,
    size: int = Path(ge=256, le=512),
    z: int = Path(ge=0),
    x: str = Path(...),
    y: str = Path(...),
) -> Response:
    """Coverage tile showing where radar data exists."""
    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")

    mode, xi, yi, lat, lon = _parse_tile_or_window_coords(x, y)
    if mode == "window":
        return await _coverage_window(request, size, z, lat, lon)

    max_tiles = 2**z
    if xi >= max_tiles or yi >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    tile_size = 512 if size >= 512 else 256

    frame = await frame_store.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No radar data available")

    # Cache the encoded coverage tile bytes keyed on the content that
    # determines them: the latest frame's timestamp + viewport.  The
    # endpoint is always PNG (fixed URL), so the format is constant and
    # needs no key element.  The "cov" namespace prefix mirrors the
    # satellite endpoint's "sat" keys - it can't collide with the radar
    # geometry/present keys (which start with the int timestamp), and it
    # deliberately sits outside ``invalidate_timestamp`` (which sweeps
    # by ``key[0] == timestamp``): coverage tracks only the latest frame,
    # so a new frame re-keys the entries and the old ones age out through
    # the LRU, same as satellite.  ``enabled_regions`` is fixed at
    # startup, so it's not in the key (the radar geometry path treats it
    # the same way).
    cache_key = ("cov", frame.timestamp, z, xi, yi, tile_size)
    cached = tile_cache.get(cache_key)
    if cached is not None:
        if isinstance(cached, CachedRender):
            tile_bytes = cached.data
            etag = cached.etag
        else:
            # Legacy raw-bytes entry (pre-ETag cache format).
            tile_bytes = cached
            etag = compute_etag(cached)
        return conditional_response(
            request=request,
            body=tile_bytes,
            etag=etag,
            content_type="image/png",
            max_age=300,
        )

    tile_bytes = await asyncio.to_thread(
        render_coverage_tile,
        frame_regions=frame.regions,
        z=z, x=xi, y=yi,
        tile_size=tile_size,
        enabled_regions=enabled_regions,
    )

    etag = compute_etag(tile_bytes)
    tile_cache.put(cache_key, CachedRender(data=tile_bytes, etag=etag))

    return conditional_response(
        request=request,
        body=tile_bytes,
        etag=etag,
        content_type="image/png",
        max_age=300,
    )


@router.get("/v2/satellite/{timestamp}/{size}/{z}/{x}/{y}/0/0_0.{ext}")
async def satellite_tile(
    request: Request,
    timestamp: int,
    size: int = Path(ge=256, le=512),
    z: int = Path(ge=0),
    x: int = Path(ge=0),
    y: int = Path(ge=0),
    ext: str = Path(pattern=r"^(png|webp)$"),
) -> Response:
    """Real satellite imagery tile, backed by NOAA GMGSI.

    Backing renderer is picked per request: the VIS-over-LW composite
    when both channels have ingested frames (the production path during
    Phase 2+), or the stand-alone LW renderer when only longwave IR is
    loaded.  When the satellite layer is disabled or neither channel has
    any frames yet, returns 503.
    """
    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")

    max_tiles = 2**z
    if x >= max_tiles or y >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    tile_size = 512 if size >= 512 else 256

    # Backing selection: composite when both channels loaded, LW-only
    # otherwise, 503 if nothing's ready.
    gmgsi_lw = satellite_grids.get("gmgsi_lw_grid") if satellite_grids else None
    gmgsi_vis = satellite_grids.get("gmgsi_vis_grid") if satellite_grids else None
    has_lw = gmgsi_lw is not None and bool(gmgsi_lw.timestamps)
    has_vis = gmgsi_vis is not None and bool(gmgsi_vis.timestamps)

    if has_lw and has_vis:
        backing = "gmgsi_composite"
    elif has_lw:
        backing = "gmgsi_lw"
    else:
        raise HTTPException(status_code=503, detail="Satellite data not available")

    if timestamp == 0:
        # Same "0 = latest" alias as the radar route: resolve before the
        # cache-key construction and max-age bucket below so alias
        # requests key and cache exactly like the canonical URL.
        timestamp = max(gmgsi_lw.timestamps)

    # Older-than-latest frames are immutable; give them a long max-age.
    # Computed before the lookup so cache hits get the same semantics.
    sat_timestamps = gmgsi_lw.timestamps
    latest_sat_ts = max(sat_timestamps) if sat_timestamps else None
    max_age = 7200 if (latest_sat_ts is not None and timestamp < latest_sat_ts) else 300

    # Distinct cache keys per backing so a runtime swap (e.g. VIS ingest
    # catching up after restart) doesn't serve stale composites.
    cache_key = ("sat", backing, timestamp, z, x, y, tile_size, ext)
    cached = tile_cache.get(cache_key)
    if cached is not None:
        if isinstance(cached, CachedRender):
            tile_bytes = cached.data
            etag = cached.etag
        else:
            # Legacy raw-bytes entry (pre-ETag cache format).
            tile_bytes = cached
            etag = compute_etag(cached)
        return conditional_response(
            request=request,
            body=tile_bytes,
            etag=etag,
            content_type=_content_type(ext),
            max_age=max_age,
            extra_headers={"X-Frame-Timestamp": str(timestamp)},
        )

    if backing == "gmgsi_composite":
        tile_bytes = await asyncio.to_thread(
            render_gmgsi_composite_tile,
            lw_source=gmgsi_lw,
            vis_source=gmgsi_vis,
            z=z, x=x, y=y,
            tile_size=tile_size,
            timestamp=timestamp,
            fmt=ext,
        )
    else:
        tile_bytes = await asyncio.to_thread(
            render_gmgsi_tile,
            source=gmgsi_lw,
            z=z, x=x, y=y,
            tile_size=tile_size,
            timestamp=timestamp,
            fmt=ext,
        )
    etag = compute_etag(tile_bytes)

    tile_cache.put(cache_key, CachedRender(data=tile_bytes, etag=etag))

    return conditional_response(
        request=request,
        body=tile_bytes,
        etag=etag,
        content_type=_content_type(ext),
        max_age=max_age,
        extra_headers={"X-Frame-Timestamp": str(timestamp)},
    )


# ---------------------------------------------------------------------------
# Alert helpers
# ---------------------------------------------------------------------------

def _parse_cap_time(value: str) -> int | None:
    """Parse CAP ISO 8601 time string to Unix epoch."""
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value).timestamp())
    except (ValueError, TypeError):
        return None


def _alert_not_expired(alert, now_utc: int) -> bool:
    """Check if alert has not expired. Returns True for alerts without expires field."""
    expires = _parse_cap_time(alert.expires)
    return expires is None or expires > now_utc


@router.get("/v2/alerts", response_model=AlertsResponse)
async def get_alerts(
    lat: float | None = Query(None, ge=-90, le=90, description="Latitude for point lookup"),
    lon: float | None = Query(None, ge=-180, le=180, description="Longitude for point lookup"),
    bbox: str | None = Query(None, description="Bounding box: west,south,east,north"),
    simplify: float = Query(1000.0, ge=0, description="Polygon simplification tolerance in meters (0=off)"),
):
    """Weather alerts as GeoJSON FeatureCollection.

    - No params: all active alerts worldwide.
    - lat+lon: alerts containing that point.  Zone-based alerts (e.g.
      Tornado Watches) are resolved to zone polygons at ingest, so every
      alert is visible in point lookups without any per-request NWS query.
    - bbox: alerts intersecting the bounding box (polygon-only).
    """
    if not alerts_enabled or alerts_store is None:
        raise HTTPException(status_code=503, detail="Alerts not available")

    alerts = alerts_store.alerts

    # Filter by point
    if lat is not None and lon is not None:
        from shapely.geometry import Point
        point = Point(lon, lat)
        alerts = [a for a in alerts if a.polygon is not None and a.polygon.intersects(point)]
    # Filter by bbox
    elif bbox is not None:
        parts = bbox.split(",")
        if len(parts) != 4:
            raise HTTPException(status_code=400, detail="bbox must be: west,south,east,north")
        try:
            w, s, e, n = map(float, parts)
        except ValueError:
            raise HTTPException(status_code=400, detail="bbox values must be numeric")
        if w < -180 or e > 180 or s < -90 or n > 90 or w > e or s > n:
            raise HTTPException(status_code=400, detail="bbox values out of range")
        from shapely.geometry import box
        bbox_poly = box(w, s, e, n)
        alerts = [a for a in alerts if a.polygon is not None and a.polygon.intersects(bbox_poly)]

    # Expiry filter
    now_utc = int(time.time())
    alerts = [a for a in alerts if _alert_not_expired(a, now_utc)]

    # Build GeoJSON features from WMO alerts
    deg_per_meter = simplify / 111_000.0 if simplify > 0 else 0.0
    from shapely.geometry import mapping
    from shapely.ops import unary_union
    features: list[GeoJSONFeature] = []

    # Multi-area CAP alerts (Bulgaria, Romania, France, ...) arrive as many
    # AlertEntry objects sharing one url, each with its own region polygon.
    # Group by url and union the group's polygons so one feature carries the
    # full footprint instead of only the first region's polygon.
    by_uri: dict[str, list] = {}
    for alert in alerts:
        by_uri.setdefault(alert.url, []).append(alert)

    for group in by_uri.values():
        first = group[0]
        polygons = [a.polygon for a in group if a.polygon is not None]
        geom = unary_union(polygons) if polygons else None
        if deg_per_meter > 0 and geom is not None:
            geom = geom.simplify(deg_per_meter, preserve_topology=True)

        regions: list[str] = []
        for a in group:
            if a.area_desc and a.area_desc not in regions:
                regions.append(a.area_desc)

        features.append(
            GeoJSONFeature(
                type="Feature",
                properties=AlertProperties(
                    title=first.event,
                    severity=first.severity,
                    time=_parse_cap_time(first.effective),
                    expires=_parse_cap_time(first.expires),
                    description=first.description,
                    regions=regions,
                    uri=first.url,
                ),
                geometry=mapping(geom) if geom is not None else None,
            )
        )

    return AlertsResponse(type="FeatureCollection", features=features)
