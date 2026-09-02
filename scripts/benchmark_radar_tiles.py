#!/usr/bin/env python3
"""Reproducible microbenchmark for the radar tile rendering hot path."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import cv2
import numpy as np
from PIL import Image

from librewxr.colors.schemes import get_lut
from librewxr.config import settings
from librewxr.native_weather import (
    blend_radar_nowcast,
    colorize_radar,
    encode_radar_png,
    sample_radar_bilinear,
)
from librewxr.tiles.coordinates import _compute_tile_pixel_latlons
from librewxr.tiles.png_palette import encode_png


def _measure(callback, iterations: int) -> dict[str, float]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        callback()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def _encode(rgba: np.ndarray, implementation: str | None) -> bytes:
    encoded = encode_radar_png(rgba, implementation=implementation)
    return encoded if encoded is not None else encode_png(Image.fromarray(rgba, "RGBA"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--size", type=int, choices=(256, 512), default=512)
    parser.add_argument(
        "--implementation", choices=("auto", "python", "rust"), default="auto",
    )
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")

    implementation = None if args.implementation == "auto" else args.implementation
    size = args.size
    rng = np.random.default_rng(20260902)
    source = rng.integers(0, 256, (1024, 1024), dtype=np.uint8)
    source[source < 24] = 0
    row = np.ascontiguousarray(
        rng.uniform(0, source.shape[0] - 1, (size, size)), dtype=np.float32,
    )
    col = np.ascontiguousarray(
        rng.uniform(0, source.shape[1] - 1, (size, size)), dtype=np.float32,
    )
    radar = sample_radar_bilinear(
        source, row, col, implementation=implementation,
    )
    model_raw = rng.integers(0, 256, radar.shape, dtype=np.uint8)
    model = cv2.GaussianBlur(model_raw.astype(np.float32), (5, 5), 0)
    model = np.ascontiguousarray(model, dtype=np.float32)
    feather = np.ascontiguousarray(rng.random(radar.shape), dtype=np.float32)
    snow_mask = np.ascontiguousarray(rng.random(radar.shape) > 0.85)
    rain_lut = get_lut(2, snow=False)
    snow_lut = get_lut(2, snow=True)
    rgba = colorize_radar(
        radar,
        rain_lut,
        snow_lut=snow_lut,
        snow_mask=snow_mask,
        display_threshold=108,
        implementation=implementation,
    )

    # Warm one-time imports/allocators before measurement.
    _encode(rgba, implementation)
    benchmarks = {
        "coordinates": _measure(
            lambda: _compute_tile_pixel_latlons(8, 132, 81, size),
            args.iterations,
        ),
        "sampling": _measure(
            lambda: sample_radar_bilinear(
                source, row, col, implementation=implementation,
            ),
            args.iterations,
        ),
        "nwp_blend": _measure(
            lambda: blend_radar_nowcast(
                radar, model, model_raw, feather, 0.4, 108,
                implementation=implementation,
            ),
            args.iterations,
        ),
        "snow_colorize": _measure(
            lambda: colorize_radar(
                radar,
                rain_lut,
                snow_lut=snow_lut,
                snow_mask=snow_mask,
                display_threshold=108,
                implementation=implementation,
            ),
            args.iterations,
        ),
        "blur": _measure(
            lambda: cv2.GaussianBlur(rgba, (5, 5), 0), args.iterations,
        ),
        "encode": _measure(
            lambda: _encode(rgba, implementation),
            args.iterations,
        ),
    }
    print(json.dumps({
        "fixture": {"seed": 20260902, "tile_size": size},
        "implementation": args.implementation,
        "configured_native_render": settings.native_render,
        "iterations": args.iterations,
        "stages": benchmarks,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
