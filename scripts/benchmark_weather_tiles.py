# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Standalone benchmark for scalar weather-tile sampling and encoding.

The benchmark deliberately has no pytest-benchmark dependency. It exercises
the real NWPChain -> ECMWF sampling -> palette -> PNG/WebP path against a
deterministic synthetic global model, and reports JSON suitable for comparing
two revisions::

    .venv/bin/python scripts/benchmark_weather_tiles.py \
        --iterations 5 --parallel 8 --output /tmp/weather-before.json

Run it on an otherwise idle machine. Absolute timings are machine-specific;
the case keys, fixtures, encoded sizes, and decoded pixels are deterministic.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

# Complete source discovery through the same deferred path used at application
# startup before importing the IFS subpackage directly. This avoids benchmark-
# only circular-import warnings from ``sources._base -> data.regions``.
import librewxr.data.regions  # noqa: F401

from librewxr.colors.weather_palettes import WEATHER_PALETTES, WeatherPalette
from librewxr.config import settings
from librewxr.data.nwp_source import NWPChain
from librewxr.data.weather_fields import WeatherField, encode_field
from librewxr.data.weather_sampling import clear_sampling_plan_cache
from librewxr.sources.world.ifs import grid as ifs_module
from librewxr.sources.world.ifs.grid import ECMWFGrid
from librewxr.sources.world.ifs.models import WeatherFrame
from librewxr.tiles.cache import CachedRender, TileCache
from librewxr.tiles.weather_renderer import (
    WEATHER_RENDERER_VERSION,
    render_scalar_weather_tile,
)


FIELD_CASES: tuple[tuple[WeatherField, str], ...] = (
    (WeatherField.TEMPERATURE_2M, "temperature"),
    (WeatherField.PRESSURE_MSL, "pressure"),
    (WeatherField.RELATIVE_HUMIDITY_2M, "humidity"),
    (WeatherField.WIND_SPEED_10M, "wind_speed"),
)
SIZES = (256, 512)
FORMATS = ("png", "webp")
ZOOMS = (3, 6, 10)
T0 = 1_700_000_000
T1 = T0 + 3_600
TIMESTAMP = T0 + 1_800


def _percentile(samples: list[float], percentile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _timed(call: Callable[[], object]) -> tuple[object, float]:
    started = time.perf_counter_ns()
    result = call()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return result, elapsed_ms


def _measure_peak(call: Callable[[], bytes]) -> tuple[bytes, float, float]:
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter_ns()
    result = call()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed_ms, peak / (1024.0 * 1024.0)


def _encoded_fields(step: float) -> dict[WeatherField, np.ndarray]:
    """Build smooth deterministic fields on a compact 0.5-degree grid."""

    latitude = np.linspace(90.0, -90.0, 361, dtype=np.float32)[:, None]
    longitude = np.linspace(-180.0, 180.0, 720, endpoint=False, dtype=np.float32)[
        None, :
    ]
    lat_radians = np.deg2rad(latitude)
    lon_radians = np.deg2rad(longitude)
    temperature = (
        14.0
        + 24.0 * np.sin(lat_radians)
        + 7.0 * np.cos(2.0 * lon_radians)
        + step * 2.0
    )
    dewpoint = temperature - (
        4.0 + 10.0 * (0.5 + 0.5 * np.cos(lat_radians + lon_radians))
    )
    pressure = (
        1005.0
        + 23.0 * np.cos(lat_radians) * np.sin(2.0 * lon_radians)
        + step * 3.0
    )
    wind_u = 8.0 * np.cos(lat_radians) * np.cos(lon_radians) + step
    wind_v = 6.0 * np.sin(lat_radians) * np.sin(2.0 * lon_radians) - step
    return {
        WeatherField.TEMPERATURE_2M: encode_field(
            WeatherField.TEMPERATURE_2M,
            np.broadcast_to(temperature, (361, 720)),
        ),
        WeatherField.DEWPOINT_2M: encode_field(
            WeatherField.DEWPOINT_2M,
            np.broadcast_to(dewpoint, (361, 720)),
        ),
        WeatherField.PRESSURE_MSL: encode_field(
            WeatherField.PRESSURE_MSL,
            np.broadcast_to(pressure, (361, 720)),
        ),
        WeatherField.WIND_U_10M: encode_field(
            WeatherField.WIND_U_10M,
            np.broadcast_to(wind_u, (361, 720)),
        ),
        WeatherField.WIND_V_10M: encode_field(
            WeatherField.WIND_V_10M,
            np.broadcast_to(wind_v, (361, 720)),
        ),
    }


def _build_source() -> tuple[ECMWFGrid, NWPChain]:
    # A compact grid keeps the benchmark cheap to set up while exercising the
    # same bilinear/temporal/derived-field code as the 0.1-degree production
    # grid. Sampling-plan keys include this geometry.
    ifs_module.PIXEL_SIZE = 0.5
    ifs_module.GRID_WIDTH = 720
    ifs_module.GRID_HEIGHT = 361
    ifs_module.GRID_SHAPE = (361, 720)
    grid = ECMWFGrid()
    grid._timesteps[T0] = WeatherFrame(T0, _encoded_fields(0.0))
    grid._timesteps[T1] = WeatherFrame(T1, _encoded_fields(1.0))
    grid._reference_time = "2023-11-14T22:13:20Z"
    grid._content_version = 1
    return grid, NWPChain([grid])


def _render(
    source: NWPChain,
    field: WeatherField,
    palette: WeatherPalette,
    size: int,
    fmt: str,
    z: int,
    x: int,
    y: int,
) -> bytes:
    return render_scalar_weather_tile(
        source=source,
        field=field,
        palette=palette,
        timestamp=TIMESTAMP,
        z=z,
        x=x,
        y=y,
        tile_size=size,
        fmt=fmt,
    )


def _cache_key(
    grid: ECMWFGrid,
    field: WeatherField,
    palette: WeatherPalette,
    size: int,
    fmt: str,
    z: int,
    x: int,
    y: int,
) -> tuple:
    return (
        "weather",
        field.value,
        grid.model_version,
        TIMESTAMP,
        z,
        x,
        y,
        size,
        palette.id,
        fmt,
        (
            settings.webp_quality
            if fmt == "webp"
            else (
                settings.weather_png_mode,
                settings.weather_png_colors,
                settings.weather_png_dither,
            )
        ),
        WEATHER_RENDERER_VERSION,
    )


def _benchmark_case(
    grid: ECMWFGrid,
    source: NWPChain,
    field: WeatherField,
    palette: WeatherPalette,
    size: int,
    fmt: str,
    z: int,
    iterations: int,
    parallel: int,
) -> dict[str, float | int | str]:
    n = 2**z
    x = n // 2
    y = n // 2
    render = lambda: _render(source, field, palette, size, fmt, z, x, y)

    clear_sampling_plan_cache()
    cold_bytes, cold_plan_ms, cold_peak_mb = _measure_peak(render)
    warm_bytes, warm_plan_ms, warm_peak_mb = _measure_peak(render)
    if cold_bytes != warm_bytes:
        raise RuntimeError("non-deterministic renderer output")

    cache = TileCache(max_mb=64)
    key = _cache_key(grid, field, palette, size, fmt, z, x, y)

    def cached_request() -> bytes:
        cached = cache.get(key)
        if isinstance(cached, CachedRender):
            return cached.data
        data = render()
        cache.put(key, CachedRender(data, "benchmark"))
        return data

    _cold_cache_bytes, cold_tile_cache_ms = _timed(cached_request)
    warm_cache_times: list[float] = []
    for _ in range(max(iterations, 1)):
        _value, elapsed = _timed(cached_request)
        warm_cache_times.append(elapsed)

    def parallel_render(index: int) -> float:
        request_x = (x + index) % n
        started = time.perf_counter_ns()
        _render(source, field, palette, size, fmt, z, request_x, y)
        return (time.perf_counter_ns() - started) / 1_000_000.0

    task_count = max(parallel, 1)
    with ThreadPoolExecutor(max_workers=task_count) as executor:
        cold_parallel = list(executor.map(parallel_render, range(task_count)))

    def parallel_cached(_index: int) -> float:
        started = time.perf_counter_ns()
        value = cache.get(key)
        if value is None:
            raise RuntimeError("warm benchmark cache unexpectedly missed")
        return (time.perf_counter_ns() - started) / 1_000_000.0

    warm_task_count = max(task_count * max(iterations, 1), task_count)
    with ThreadPoolExecutor(max_workers=task_count) as executor:
        warm_parallel = list(executor.map(parallel_cached, range(warm_task_count)))

    return {
        "field": field.value,
        "palette": palette.id,
        "size": size,
        "format": fmt,
        "z": z,
        "encoded_bytes": len(cold_bytes),
        "cold_plan_render_ms": round(cold_plan_ms, 3),
        "warm_plan_render_ms": round(warm_plan_ms, 3),
        "cold_tile_cache_ms": round(cold_tile_cache_ms, 3),
        "warm_tile_cache_p50_ms": round(_percentile(warm_cache_times, 0.50), 4),
        "warm_tile_cache_p95_ms": round(_percentile(warm_cache_times, 0.95), 4),
        "parallel_cold_p50_ms": round(_percentile(cold_parallel, 0.50), 3),
        "parallel_cold_p95_ms": round(_percentile(cold_parallel, 0.95), 3),
        "parallel_warm_p50_ms": round(_percentile(warm_parallel, 0.50), 4),
        "parallel_warm_p95_ms": round(_percentile(warm_parallel, 0.95), 4),
        "cold_peak_tracemalloc_mb": round(cold_peak_mb, 3),
        "warm_peak_tracemalloc_mb": round(warm_peak_mb, 3),
    }


def run_benchmark(iterations: int, parallel: int) -> dict:
    grid, source = _build_source()
    cases = []
    started = time.time()
    try:
        for field, palette_id in FIELD_CASES:
            palette = WEATHER_PALETTES[palette_id]
            for size in SIZES:
                for fmt in FORMATS:
                    for z in ZOOMS:
                        cases.append(
                            _benchmark_case(
                                grid,
                                source,
                                field,
                                palette,
                                size,
                                fmt,
                                z,
                                iterations,
                                parallel,
                            )
                        )
    finally:
        import asyncio

        asyncio.run(grid.close())
        clear_sampling_plan_cache()
    return {
        "schema_version": 1,
        "iterations": iterations,
        "parallel": parallel,
        "case_count": len(cases),
        "elapsed_seconds": round(time.time() - started, 3),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(args.iterations, args.parallel)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
