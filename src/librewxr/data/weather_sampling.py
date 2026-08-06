# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Reusable spatial sampling plans for regular global weather grids."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from librewxr.config import settings


@dataclass(frozen=True)
class SamplingPlan:
    """Grid indexes and bilinear weights for one output coordinate raster.

    A plan contains geometry only. It deliberately has no frame, memmap, field,
    or timestamp reference and is therefore safe to reuse after model updates.
    """

    r0: np.ndarray
    r1: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    dr: np.ndarray
    dc: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        shape = self.r0.shape
        arrays = (self.r1, self.c0, self.c1, self.dr, self.dc, self.valid)
        if any(array.shape != shape for array in arrays):
            raise ValueError("all SamplingPlan arrays must have the same shape")
        for array in (self.r0, self.r1, self.c0, self.c1, self.dr, self.dc):
            array.flags.writeable = False
        self.valid.flags.writeable = False

    @property
    def shape(self) -> tuple[int, ...]:
        return self.valid.shape

    @property
    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.r0,
                self.r1,
                self.c0,
                self.c1,
                self.dr,
                self.dc,
                self.valid,
            )
        )


def build_regular_sampling_plan(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    west: float,
    north: float,
    pixel_size_x: float,
    pixel_size_y: float,
    width: int,
    height: int,
    wrap_longitude: bool = True,
) -> SamplingPlan:
    """Build a plan from output lat/lon arrays without touching source data."""

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.shape != lon.shape:
        raise ValueError("lat and lon must have identical shapes")
    if width <= 0 or height <= 0 or pixel_size_x <= 0 or pixel_size_y <= 0:
        raise ValueError("regular grid dimensions and pixel sizes must be positive")

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= -90.0)
        & (lat <= 90.0)
    )
    safe_lat = np.where(valid, lat, north)
    safe_lon = np.where(valid, lon, west)
    longitude_span = pixel_size_x * width
    if wrap_longitude:
        normalized_lon = np.mod(safe_lon - west, longitude_span) + west
    else:
        normalized_lon = safe_lon
        valid &= (lon >= west) & (lon <= west + longitude_span)

    row_f = (north - safe_lat) / pixel_size_y
    col_f = (normalized_lon - west) / pixel_size_x
    row_floor = np.floor(np.where(valid, row_f, 0.0))
    col_floor = np.floor(np.where(valid, col_f, 0.0))

    r0_raw = row_floor.astype(np.int32)
    r1_raw = r0_raw + 1
    r0 = np.clip(r0_raw, 0, height - 1).astype(np.int32, copy=False)
    r1 = np.clip(r1_raw, 0, height - 1).astype(np.int32, copy=False)
    if wrap_longitude:
        c0 = np.mod(col_floor.astype(np.int64), width).astype(np.int32)
        c1 = np.mod(c0.astype(np.int64) + 1, width).astype(np.int32)
    else:
        c0 = np.clip(col_floor.astype(np.int32), 0, width - 1)
        c1 = np.clip(c0 + 1, 0, width - 1)

    dr = np.clip(row_f - row_floor, 0.0, 1.0).astype(np.float32)
    dc = np.clip(col_f - col_floor, 0.0, 1.0).astype(np.float32)
    dr[~valid] = 0.0
    dc[~valid] = 0.0
    return SamplingPlan(
        r0=r0,
        r1=r1,
        c0=c0,
        c1=c1,
        dr=dr,
        dc=dc,
        valid=valid.astype(bool, copy=False),
    )


def web_mercator_tile_latlons(
    z: int,
    x: int,
    y: int,
    tile_size: int,
    padding: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel-centre coordinates for a valid XYZ Web Mercator tile."""

    if z < 0:
        raise ValueError("z must be non-negative")
    n = 2**z
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError(f"tile ({z}, {x}, {y}) is outside the XYZ pyramid")
    if tile_size <= 0 or padding < 0:
        raise ValueError("tile_size must be positive and padding non-negative")

    pixels = np.arange(-padding, tile_size + padding, dtype=np.float64) + 0.5
    lon = (x + pixels / tile_size) / n * 360.0 - 180.0
    mercator_y = 1.0 - 2.0 * (y + pixels / tile_size) / n
    lat = np.degrees(np.arctan(np.sinh(math.pi * mercator_y)))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lat_grid, lon_grid


@lru_cache(maxsize=settings.coord_cache_size)
def cached_regular_tile_sampling_plan(
    source_identity: str,
    grid_version: int,
    z: int,
    x: int,
    y: int,
    tile_size: int,
    padding: int,
    projection: str,
    west: float,
    north: float,
    pixel_size_x: float,
    pixel_size_y: float,
    width: int,
    height: int,
    wrap_longitude: bool = True,
) -> SamplingPlan:
    """Return a cached plan keyed by source, geometry, tile, and projection."""

    # ``source_identity``, ``grid_version``, and ``projection`` are intentionally
    # part of the cache key even though regular lat/lon math does not otherwise
    # consume them. They prevent accidental cross-source or post-migration reuse.
    del source_identity, grid_version
    if projection != "regular_latlon":
        raise ValueError(f"unsupported sampling projection: {projection}")
    lat, lon = web_mercator_tile_latlons(z, x, y, tile_size, padding)
    return build_regular_sampling_plan(
        lat,
        lon,
        west=west,
        north=north,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        width=width,
        height=height,
        wrap_longitude=wrap_longitude,
    )


def clear_sampling_plan_cache() -> None:
    cached_regular_tile_sampling_plan.cache_clear()
