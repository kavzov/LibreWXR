# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import functools
import json
import logging
import time
import psutil
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, HTTPException, Path, Query, Request, Response

from datetime import datetime

from librewxr.api.models import (
    AlertProperties,
    AlertsResponse,
    ColorScheme,
    GeoJSONFeature,
    RadarData,
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
from librewxr.data.point_nowcast import build_point_nowcast
from librewxr.data.store import FrameStore
from librewxr.data.weather_fields import field_spec
from librewxr.mcp.discovery import build_ai_catalog
from librewxr.memory import detect_memory_limit_mb
from librewxr.tiles.cache import CachedRender, TileCache
from librewxr.tiles.coordinates import coord_cache_bytes, coord_cache_stats
from librewxr.tiles.renderer import (
    compute_tile_geometry,
    present_tile,
    render_coverage_tile,
)
from librewxr.tiles.request_tracker import TileRequestTracker
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

# Tile present pool - set by main.py.  Multi-mode render workers get a
# dedicated executor for the cheap ``present_tile`` tail (colorize/encode)
# so those jobs never queue behind long geometry computes on the shared
# default executor under a cold-tile burst.  Single mode leaves this None
# and the tile endpoints fall back to ``asyncio.to_thread`` (the loop
# default executor), byte-identical to the pre-split behaviour.
present_executor = None  # ThreadPoolExecutor | None

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


async def _weather_tile_singleflight(
    key: tuple,
    factory: Callable[[], Awaitable[CachedRender]],
) -> CachedRender:
    """Share one cold weather render among concurrent identical requests."""

    task = _weather_tile_flights.get(key)
    if task is None:
        task = asyncio.create_task(factory())
        _weather_tile_flights[key] = task

        def _remove(done: asyncio.Task[CachedRender]) -> None:
            if _weather_tile_flights.get(key) is done:
                _weather_tile_flights.pop(key, None)

        task.add_done_callback(_remove)
    # A cancelled client must not cancel the shared render needed by peers.
    return await asyncio.shield(task)


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
        "other_mb": round(other_bytes / (1024 * 1024), 1),
    })

    # Split the tile cache into its three entry kinds: satellite render
    # entries (``"sat"``-prefixed keys), geometry entries (int timestamp +
    # 6-element viewport key), and present render entries (int timestamp +
    # 9-element viewport/visual key).  Each kind is reported with its own
    # count and byte total.
    cache_kind_geometry = 0
    cache_kind_geometry_bytes = 0
    cache_kind_present = 0
    cache_kind_present_bytes = 0
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

    return {
        "status": "ok" if frame_count > 0 else "degraded",
        "uptime_seconds": uptime,
        "memory": {
            "resident_mb": round(rss_mb, 1),
            "limit_mb": round(mem_limit_mb, 1),
            "usage_pct": ram_usage,
            "breakdown": breakdown,
        },
        "frames": {
            "count": frame_count,
            "max": settings.max_frames,
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
) -> Response:
    """Render a continuous global weather field without changing radar APIs."""

    weather_field = PUBLIC_WEATHER_FIELDS.get(field)
    if weather_field is None:
        raise HTTPException(status_code=404, detail=f"Unknown weather field: {field}")
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
    host = settings.public_url.rstrip("/")

    past = [
        RadarTimestamp(time=ts, path=f"/v2/radar/{ts}")
        for ts in sorted(timestamps)
    ]

    nowcast = []
    if nowcast_store is not None:
        nc_timestamps = await nowcast_store.get_timestamps()
        nowcast = [
            RadarTimestamp(time=ts, path=f"/v2/radar/{ts}")
            for ts in nc_timestamps
        ]

    infrared = []
    # Catalog timestamps come from GMGSI LW since LW is the always-on
    # 24/7 baseline (VIS only carries the daytime half of the day).
    # When the satellite layer is disabled or unloaded the array is
    # empty and the tile endpoint returns 503.
    gmgsi_lw = satellite_grids.get("gmgsi_lw_grid") if satellite_grids else None
    if gmgsi_lw is not None and gmgsi_lw.timestamps:
        infrared = [
            RadarTimestamp(time=ts, path=f"/v2/satellite/{ts}")
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
        radar=RadarData(past=past, nowcast=nowcast, colorSchemes=color_schemes),
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


async def _present_tile_async(geom, **kwargs) -> bytes:
    """Run ``present_tile`` off the event loop.

    Multi-mode render workers get a dedicated present pool
    (``routes.present_executor``) so cheap colorize/encode jobs never queue
    behind long geometry computes on the shared default executor.  Single
    mode leaves ``present_executor`` None and falls back to
    ``asyncio.to_thread`` - byte-identical to the pre-split behaviour.
    """
    if present_executor is not None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            present_executor,
            functools.partial(present_tile, geom, **kwargs),
        )
    return await asyncio.to_thread(present_tile, geom, **kwargs)


@router.get("/v2/radar/{timestamp}/{size}/{z}/{x}/{y}/{color}/{smooth_snow}.{ext}")
async def radar_tile(
    request: Request,
    timestamp: int,
    size: int = Path(ge=256, le=512),
    z: int = Path(ge=0),
    x: int = Path(ge=0),
    y: int = Path(ge=0),
    color: int = Path(ge=0, le=255),
    smooth_snow: str = Path(pattern=r"^\d+_\d+$"),
    ext: str = Path(pattern=r"^(png|webp)$"),
    arrows: str = Query(default=""),
    cells: str = Query(default=""),
    min_dbz: float | None = Query(default=None, ge=-32.0, le=95.5),
) -> Response:
    """Rain Viewer-compatible tile endpoint."""
    logger.debug("Tile request: z=%d x=%d y=%d color=%d smooth_snow=%s ext=%s", z, x, y, color, smooth_snow, ext)
    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")

    max_tiles = 2**z
    if x >= max_tiles or y >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    if tile_request_tracker is not None:
        tile_request_tracker.record(z, x, y)

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

    # Geometry cache: keyed only on inputs that affect the sampled values
    # (radar source + viewport + smoothing + snow-mask presence).  Color
    # scheme, output format, and arrow style apply per-request in
    # ``present_tile`` so a single cached entry serves every visual
    # variant of the same viewport.
    geom_key = (timestamp, z, x, y, tile_size, smooth, snow)
    geom = tile_cache.get(geom_key)

    # Geometry-stage cache outcome: this is the meaningful hit/miss for
    # "is the fast path helping" — a transparent geometry that comes back
    # from the cache was paid for on an earlier request, so only the
    # miss-side fast-path counter below attributes the work to this one.
    if tile_request_tracker is not None:
        if geom is not None:
            tile_request_tracker.record_cache_hit()
        else:
            tile_request_tracker.record_cache_miss()

    # We need the radar frame whenever geometry must be computed AND
    # whenever arrows are requested (arrow rendering needs live frame
    # data + flow fields).  Skip the fetch on pure cache hits without
    # arrows — that's the hot path Merry Sky-style clients exercise.
    frame = None
    nowcast_blend = None
    is_nowcast = False
    need_frame = geom is None or bool(arrow_style)
    if need_frame:
        frame = await frame_store.get_frame(timestamp)
        if frame is None and nowcast_store is not None:
            nc_frame, nowcast_blend = await nowcast_store.get_frame(timestamp)
            if nc_frame is not None:
                frame = nc_frame
                is_nowcast = True
        if frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")

    if geom is None:
        geom = await asyncio.to_thread(
            compute_tile_geometry,
            frame_regions=frame.regions,
            z=z, x=x, y=y,
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
        # Only fire on the cold-compute path: a fast-path label here means
        # this request actually paid for the empty-tile work (cache hits
        # of a previously-computed transparent geometry are already counted
        # by ``record_cache_hit`` above, not a fast-path firing now).
        if tile_request_tracker is not None and geom.fast_path is not None:
            tile_request_tracker.record_fast_path(geom.fast_path)

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

    is_plain = not arrows and not cells

    if is_plain:
        # Present-stage cache: one entry per visual variant of the same
        # geometry.  Stores the encoded bytes plus the ETag so a present
        # cache hit skips both ``present_tile`` and the ETag hash.
        present_key = (
            timestamp, z, x, y, tile_size, smooth, snow,
            color, ext, settings.webp_quality, display_min_dbz,
        )
        cached = tile_cache.get(present_key)
        if isinstance(cached, CachedRender):
            tile_bytes = cached.data
            etag = cached.etag
        else:
            tile_bytes = await _present_tile_async(
                geom,
                color_scheme=color,
                fmt=ext,
                display_min_dbz=display_min_dbz,
                arrow_style=arrow_style if (flow_regions or nwp_flow is not None) else "",
                flow_regions=flow_regions,
                frame_regions=frame.regions if frame is not None else None,
                enabled_regions=enabled_regions,
                nwp_flow=nwp_flow,
                nwp_chain=nwp_chain,
                frame_timestamp=timestamp,
                z=z, x=x, y=y,
                cell_style=cell_style if cells_by_region else "",
                cells_by_region=cells_by_region,
                cell_counts=cell_counts,
            )
            etag = compute_etag(tile_bytes)
            tile_cache.put(present_key, CachedRender(data=tile_bytes, etag=etag))
    else:
        # Overlay requests (arrows / cells) evolve under the same
        # timestamp, so their rendered bytes are never cached.
        tile_bytes = await _present_tile_async(
            geom,
            color_scheme=color,
            fmt=ext,
            display_min_dbz=display_min_dbz,
            arrow_style=arrow_style if (flow_regions or nwp_flow is not None) else "",
            flow_regions=flow_regions,
            frame_regions=frame.regions if frame is not None else None,
            enabled_regions=enabled_regions,
            nwp_flow=nwp_flow,
            nwp_chain=nwp_chain,
            frame_timestamp=timestamp,
            z=z, x=x, y=y,
            cell_style=cell_style if cells_by_region else "",
            cells_by_region=cells_by_region,
            cell_counts=cell_counts,
        )
        etag = compute_etag(tile_bytes)

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
                z=z, x=x, y=y,
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

    return conditional_response(
        request=request,
        body=tile_bytes,
        etag=etag,
        content_type=_content_type(ext),
        max_age=max_age,
    )


@router.get("/v2/coverage/0/{size}/{z}/{x}/{y}/0/0_0.png")
async def coverage_tile(
    request: Request,
    size: int = Path(ge=256, le=512),
    z: int = Path(ge=0),
    x: int = Path(ge=0),
    y: int = Path(ge=0),
) -> Response:
    """Coverage tile showing where radar data exists."""
    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")

    max_tiles = 2**z
    if x >= max_tiles or y >= max_tiles:
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
    cache_key = ("cov", frame.timestamp, z, x, y, tile_size)
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
        z=z, x=x, y=y,
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
    features: list[GeoJSONFeature] = []
    seen_uris: set[str] = set()

    for alert in alerts:
        geom = alert.polygon
        if deg_per_meter > 0 and geom is not None:
            geom = geom.simplify(deg_per_meter, preserve_topology=True)

        uri = alert.url
        if uri in seen_uris:
            continue
        seen_uris.add(uri)

        features.append(
            GeoJSONFeature(
                type="Feature",
                properties=AlertProperties(
                    title=alert.event,
                    severity=alert.severity,
                    time=_parse_cap_time(alert.effective),
                    expires=_parse_cap_time(alert.expires),
                    description=alert.description,
                    regions=[alert.area_desc] if alert.area_desc else [],
                    uri=uri,
                ),
                geometry=mapping(geom) if geom is not None else None,
            )
        )

    return AlertsResponse(type="FeatureCollection", features=features)
