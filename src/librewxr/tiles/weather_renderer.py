# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Renderer for continuous scalar weather fields."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from librewxr.colors.weather_palettes import WeatherPalette
from librewxr.config import settings
from librewxr.data.weather_fields import WeatherField
from librewxr.tiles.png_palette import encode_png

WEATHER_RENDERER_VERSION = 2


def colorize_weather_values(
    values: np.ndarray,
    palette: WeatherPalette,
) -> np.ndarray:
    """Normalize physical values and colorize them through a prebuilt LUT."""

    physical = np.asarray(values, dtype=np.float32)
    # One reusable float32 scratch replaces three persistent boolean masks,
    # the gathered inside-values vector, and an int32 index raster. uint16 is
    # sufficient for the fixed 4096-entry LUT.
    scaled = np.empty(physical.shape, dtype=np.float32)
    np.subtract(physical, np.float32(palette.minimum), out=scaled)
    scaled /= np.float32(palette.maximum - palette.minimum)
    scaled *= np.float32(len(palette.lut) - 1)
    np.nan_to_num(scaled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(scaled, 0.0, len(palette.lut) - 1, out=scaled)
    np.rint(scaled, out=scaled)
    indices = scaled.astype(np.uint16)
    rgba = palette.lut[indices]

    mask = physical < palette.minimum
    rgba[mask] = palette.below_rgba
    mask = physical > palette.maximum
    rgba[mask] = palette.above_rgba
    mask = ~np.isfinite(physical)
    rgba[mask] = palette.nodata_rgba
    return rgba


def encode_weather_image(rgba: np.ndarray, fmt: str) -> bytes:
    """Encode an RGBA weather tile as PNG or WebP without smoothing."""

    image = Image.fromarray(rgba, "RGBA")
    if fmt == "webp":
        buffer = io.BytesIO()
        if settings.webp_quality >= 100:
            image.save(buffer, format="WEBP", lossless=True, method=1)
        else:
            image.save(buffer, format="WEBP", quality=settings.webp_quality)
        return buffer.getvalue()
    if fmt != "png":
        raise ValueError(f"unsupported weather tile format: {fmt}")
    return encode_png(
        image,
        quantize=settings.weather_png_mode == "quantized",
        colors=settings.weather_png_colors,
        dither=settings.weather_png_dither,
    )


def render_scalar_weather_tile(
    source,
    field: WeatherField,
    palette: WeatherPalette,
    timestamp: int,
    z: int,
    x: int,
    y: int,
    tile_size: int,
    fmt: str,
) -> bytes:
    """Sample one physical field tile, LUT-colorize it, and encode it."""

    normalized = WeatherField(field)
    if palette.field is not normalized:
        raise ValueError(
            f"palette {palette.id} does not support {normalized.value}"
        )
    values = np.asarray(
        source.sample_tile_field(
            normalized,
            z,
            x,
            y,
            timestamp=timestamp,
            tile_size=tile_size,
            padding=0,
            bilinear=True,
        ),
        dtype=np.float32,
    )
    expected_shape = (tile_size, tile_size)
    if values.shape != expected_shape:
        raise ValueError(
            f"weather source returned {values.shape}, expected {expected_shape}"
        )
    return encode_weather_image(colorize_weather_values(values, palette), fmt)
