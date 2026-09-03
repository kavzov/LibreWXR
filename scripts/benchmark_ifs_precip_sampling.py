#!/usr/bin/env python3
"""Compare legacy coordinate sampling with the cached XYZ precipitation plan."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time

import numpy as np

logging.disable(logging.CRITICAL)

from librewxr.data.weather_fields import WeatherField
from librewxr.data.weather_sampling import (
    clear_sampling_plan_cache,
    web_mercator_tile_latlons,
)
from librewxr.sources.world.ifs.grid import ECMWFGrid, GRID_SHAPE
from librewxr.sources.world.ifs.models import WeatherFrame


def measure(call, iterations: int) -> dict[str, float]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(samples)
    return {
        "mean_ms": round(statistics.mean(samples), 3),
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--zoom", type=int, default=8)
    parser.add_argument("--x", type=int, default=131)
    parser.add_argument("--y", type=int, default=83)
    parser.add_argument("--padding", type=int, default=3)
    parser.add_argument("--bilinear", action="store_true")
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")

    rng = np.random.default_rng(20260903)
    precipitation = rng.integers(0, 256, GRID_SHAPE, dtype=np.uint8)
    precipitation[precipitation < 32] = 0
    timestamp = 1_788_460_000
    grid = ECMWFGrid()
    grid._timesteps[timestamp] = WeatherFrame(
        timestamp,
        {WeatherField.PRECIPITATION: precipitation},
        np.zeros(GRID_SHAPE, dtype=bool),
    )
    grid._sorted_timestamps = [timestamp]

    def legacy():
        lat, lon = web_mercator_tile_latlons(
            args.zoom, args.x, args.y, args.tile_size, args.padding
        )
        return grid.sample(lat, lon, timestamp, args.bilinear)

    def planned():
        return grid.sample_tile(
            args.zoom,
            args.x,
            args.y,
            timestamp,
            args.tile_size,
            args.padding,
            args.bilinear,
        )

    expected = legacy()
    clear_sampling_plan_cache()
    cold_started = time.perf_counter_ns()
    actual = planned()
    cold_ms = (time.perf_counter_ns() - cold_started) / 1_000_000
    np.testing.assert_array_equal(actual, expected)
    planned()  # Ensure the timed plan path is a cache hit.
    legacy_result = measure(legacy, args.iterations)
    planned_result = measure(planned, args.iterations)
    print(json.dumps({
        "tile_size": args.tile_size,
        "padding": args.padding,
        "bilinear": args.bilinear,
        "iterations": args.iterations,
        "cold_plan_ms": round(cold_ms, 3),
        "legacy": legacy_result,
        "planned": planned_result,
        "warm_speedup": round(
            legacy_result["mean_ms"] / planned_result["mean_ms"], 2
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
