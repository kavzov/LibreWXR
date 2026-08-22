# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Small-area radar sampling for the public point-nowcast API.

The tile API presents colourised images, while this module samples the
underlying uint8 dBZ grids directly.  Keeping the calculation below the API
layer avoids reverse-mapping palette colours and gives callers an explicit
coverage signal when a coordinate cannot be resolved to an enabled radar
region.
"""

import math
import time
from dataclasses import dataclass

import numpy as np

from librewxr.data.regions import RegionDef
from librewxr.mcp.sampling import dbz_to_rate_mmh, resolve_region_for_point
from librewxr.tiles.coordinates import _laea_forward, _tmerc_forward


@dataclass(frozen=True)
class RadarNeighborhoodSample:
    """Summary of the radar pixels inside a circular neighbourhood."""

    coverage: str
    region: str | None
    sample_count: int
    wet_pixel_count: int
    wet_fraction: float | None
    max_dbz: float | None
    max_rate_mmh: float | None


def _point_pixel_coordinates(
    region: RegionDef,
    lat: float,
    lon: float,
) -> tuple[float, float]:
    """Return fractional ``(row, column)`` coordinates in *region*."""

    if region.proj == "latlon":
        col = (lon - region.west) / region.pixel_size
        row = (region.north - lat) / region._ps_y
        return row, col
    if region.proj == "laea":
        x, y = _laea_forward(
            np.asarray([lon], dtype=np.float64),
            np.asarray([lat], dtype=np.float64),
            region,
        )
    elif region.proj == "tmerc":
        x, y = _tmerc_forward(
            np.asarray([lon], dtype=np.float64),
            np.asarray([lat], dtype=np.float64),
            region,
        )
    else:
        raise ValueError(
            f"Unknown projection '{region.proj}' for region '{region.name}'"
        )
    col = (float(x[0]) - region.grid_x_min) / region.grid_scale
    row = (region.grid_y_max - float(y[0])) / region.grid_scale
    return row, col


def _pixel_size_km(region: RegionDef, lat: float) -> tuple[float, float]:
    """Return approximate ``(row_km, column_km)`` grid spacing."""

    if region.proj in {"laea", "tmerc"}:
        spacing = region.grid_scale / 1000.0
        return spacing, spacing
    row_km = region._ps_y * 111.32
    # Avoid a zero longitudinal distance at the exact poles.  Radar regions
    # do not reach them, but the public endpoint accepts the full WGS84 range.
    col_km = region.pixel_size * 111.32 * max(abs(math.cos(math.radians(lat))), 1e-3)
    return row_km, col_km


def sample_neighborhood(
    region: RegionDef,
    lat: float,
    lon: float,
    radius_km: float,
    frame_array: np.ndarray,
    noise_floor_dbz: float,
) -> RadarNeighborhoodSample:
    """Summarise pixels within *radius_km* of a geographic point.

    Distances are measured in the native grid around the query point.  The
    approximation is sub-pixel at the small (1-10 km) radii accepted by the
    API and works for regular lat/lon, LAEA, and transverse-Mercator regions.
    """

    row_f, col_f = _point_pixel_coordinates(region, lat, lon)
    row_km, col_km = _pixel_size_km(region, lat)
    if row_km <= 0.0 or col_km <= 0.0:
        return RadarNeighborhoodSample(
            coverage="out_of_range",
            region=region.name,
            sample_count=0,
            wet_pixel_count=0,
            wet_fraction=None,
            max_dbz=None,
            max_rate_mmh=None,
        )

    row_radius = int(math.ceil(radius_km / row_km))
    col_radius = int(math.ceil(radius_km / col_km))
    row_min = max(0, int(math.floor(row_f)) - row_radius)
    row_max = min(frame_array.shape[0] - 1, int(math.ceil(row_f)) + row_radius)
    col_min = max(0, int(math.floor(col_f)) - col_radius)
    col_max = min(frame_array.shape[1] - 1, int(math.ceil(col_f)) + col_radius)
    if row_min > row_max or col_min > col_max:
        return RadarNeighborhoodSample(
            coverage="out_of_range",
            region=region.name,
            sample_count=0,
            wet_pixel_count=0,
            wet_fraction=None,
            max_dbz=None,
            max_rate_mmh=None,
        )

    rows = np.arange(row_min, row_max + 1, dtype=np.float64)
    cols = np.arange(col_min, col_max + 1, dtype=np.float64)
    row_dist = (rows[:, None] - row_f) * row_km
    col_dist = (cols[None, :] - col_f) * col_km
    circle = row_dist * row_dist + col_dist * col_dist <= radius_km * radius_km
    pixels = np.asarray(frame_array[row_min:row_max + 1, col_min:col_max + 1])[circle]
    if pixels.size == 0:
        # A radius smaller than half a native pixel can otherwise miss every
        # pixel centre.  Sampling the nearest pixel is the least surprising
        # representation of that point-sized query.
        row = int(round(row_f))
        col = int(round(col_f))
        if row < 0 or row >= frame_array.shape[0] or col < 0 or col >= frame_array.shape[1]:
            return RadarNeighborhoodSample(
                coverage="out_of_range",
                region=region.name,
                sample_count=0,
                wet_pixel_count=0,
                wet_fraction=None,
                max_dbz=None,
                max_rate_mmh=None,
            )
        pixels = np.asarray([frame_array[row, col]], dtype=np.uint8)

    threshold = max(1, int(math.ceil((noise_floor_dbz + 32.0) * 2.0)))
    wet = pixels >= threshold
    wet_count = int(np.count_nonzero(wet))
    sample_count = int(pixels.size)
    if wet_count:
        max_pixel = int(np.max(pixels[wet]))
        max_dbz = max_pixel / 2.0 - 32.0
        max_rate_mmh = dbz_to_rate_mmh(max_dbz)
    else:
        max_dbz = None
        max_rate_mmh = None
    return RadarNeighborhoodSample(
        coverage="in_range",
        region=region.name,
        sample_count=sample_count,
        wet_pixel_count=wet_count,
        wet_fraction=wet_count / sample_count,
        max_dbz=max_dbz,
        max_rate_mmh=max_rate_mmh,
    )


async def build_point_nowcast(
    frame_store,
    nowcast_store,
    enabled_regions: list[str],
    lat: float,
    lon: float,
    radius_km: float,
    past_minutes: int,
    future_minutes: int,
    noise_floor_dbz: float,
    fetch_interval: int,
) -> dict:
    """Build a bounded observed + forecast radar series for one point."""

    timestamps = await frame_store.get_timestamps() if frame_store is not None else []
    if not timestamps:
        raise LookupError("Radar observations not available")

    latest_timestamp = max(timestamps)
    region = resolve_region_for_point(lat, lon, enabled_regions)
    frames: list[dict] = []

    for timestamp in sorted(timestamps):
        offset_seconds = timestamp - latest_timestamp
        if offset_seconds < -past_minutes * 60:
            continue
        frame = await frame_store.get_frame(timestamp)
        sample = _sample_frame(
            region, lat, lon, radius_km, frame, noise_floor_dbz,
        )
        frames.append(
            _frame_payload(timestamp, offset_seconds, "observed", sample, 1.0)
        )

    if nowcast_store is not None:
        for timestamp in await nowcast_store.get_timestamps():
            offset_seconds = timestamp - latest_timestamp
            if offset_seconds <= 0 or offset_seconds > future_minutes * 60:
                continue
            frame, blend_weight = await nowcast_store.get_frame(timestamp)
            sample = _sample_frame(
                region, lat, lon, radius_km, frame, noise_floor_dbz,
            )
            frames.append(
                _frame_payload(
                    timestamp,
                    offset_seconds,
                    "forecast",
                    sample,
                    float(blend_weight),
                )
            )

    frames.sort(key=lambda item: item["time"])
    now = int(time.time())
    latest_age_seconds = max(0, now - latest_timestamp)
    forecast_offsets = [
        frame["minutes_offset"] for frame in frames if frame["period"] == "forecast"
    ]
    history_offsets = [
        -frame["minutes_offset"] for frame in frames if frame["period"] == "observed"
    ]
    return {
        "generated": now,
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius_km,
        "noise_floor_dbz": noise_floor_dbz,
        "latest_observation_time": latest_timestamp,
        "latest_age_seconds": latest_age_seconds,
        "stale": latest_age_seconds > max(fetch_interval * 2, 600),
        "history_minutes_available": max(history_offsets, default=0),
        "forecast_minutes_available": max(forecast_offsets, default=0),
        "frames": frames,
    }


def _sample_frame(
    region: RegionDef | None,
    lat: float,
    lon: float,
    radius_km: float,
    frame,
    noise_floor_dbz: float,
) -> RadarNeighborhoodSample:
    if region is None or frame is None:
        return RadarNeighborhoodSample(
            coverage="out_of_range",
            region=region.name if region is not None else None,
            sample_count=0,
            wet_pixel_count=0,
            wet_fraction=None,
            max_dbz=None,
            max_rate_mmh=None,
        )
    array = frame.regions.get(region.name)
    if array is None:
        return RadarNeighborhoodSample(
            coverage="out_of_range",
            region=region.name,
            sample_count=0,
            wet_pixel_count=0,
            wet_fraction=None,
            max_dbz=None,
            max_rate_mmh=None,
        )
    return sample_neighborhood(
        region, lat, lon, radius_km, array, noise_floor_dbz,
    )


def _frame_payload(
    timestamp: int,
    offset_seconds: int,
    period: str,
    sample: RadarNeighborhoodSample,
    blend_weight: float,
) -> dict:
    return {
        "time": int(timestamp),
        "minutes_offset": int(round(offset_seconds / 60.0)),
        "period": period,
        "coverage": sample.coverage,
        "region": sample.region,
        "sample_count": sample.sample_count,
        "wet_pixel_count": sample.wet_pixel_count,
        "wet_fraction": sample.wet_fraction,
        "max_dbz": sample.max_dbz,
        "max_rate_mmh": sample.max_rate_mmh,
        "blend_weight": blend_weight,
    }
