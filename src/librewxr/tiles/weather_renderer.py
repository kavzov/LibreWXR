# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Renderer for continuous scalar weather fields."""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageDraw

from librewxr.colors.weather_palettes import WeatherPalette
from librewxr.config import settings
from librewxr.data.weather_fields import WeatherField
from librewxr.tiles.png_palette import encode_png

WEATHER_RENDERER_VERSION = 4


def sample_scalar_weather_point(
    source,
    field: WeatherField,
    timestamp: int,
    latitude: float,
    longitude: float,
) -> float | None:
    """Sample one point through the same physical-value path as weather tiles."""

    values = np.asarray(
        source.sample_field(
            WeatherField(field),
            np.asarray([latitude], dtype=np.float64),
            np.asarray([longitude], dtype=np.float64),
            timestamp=timestamp,
            bilinear=True,
        ),
        dtype=np.float32,
    )
    if values.shape != (1,):
        raise ValueError(f"weather source returned {values.shape}, expected (1,)")
    value = float(values[0])
    return value if np.isfinite(value) else None


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


def draw_wind_vectors(
    rgba: np.ndarray,
    u_values: np.ndarray,
    v_values: np.ndarray,
    style: str,
) -> np.ndarray:
    """Overlay compact arrows showing the direction air is moving toward."""

    if style not in ("light", "dark"):
        raise ValueError(f"unsupported wind vector style: {style}")
    height, width = rgba.shape[:2]
    if u_values.shape != (height, width) or v_values.shape != (height, width):
        raise ValueError("wind vector components must match the RGBA tile shape")

    image = Image.fromarray(rgba, "RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    color = (36, 44, 55, 185) if style == "dark" else (255, 255, 255, 190)
    spacing = 40 if width <= 256 else 56
    line_width = 1 if width <= 256 else 2
    maximum_length = 10.0 if width <= 256 else 14.0
    head_length = 3.0 if width <= 256 else 4.0
    head_width = 2.0 if width <= 256 else 3.0

    for y in range(spacing // 2, height, spacing):
        for x in range(spacing // 2, width, spacing):
            u = float(u_values[y, x])
            v = float(v_values[y, x])
            if not math.isfinite(u) or not math.isfinite(v):
                continue
            speed = math.hypot(u, v)
            if speed < 0.5:
                continue

            length = min(maximum_length, 4.0 + speed * 0.35)
            unit_x = u / speed
            unit_y = -v / speed  # northward wind points up on a map tile
            half_x = unit_x * length / 2.0
            half_y = unit_y * length / 2.0
            tail = (x - half_x, y - half_y)
            tip = (x + half_x, y + half_y)
            draw.line((tail, tip), fill=color, width=line_width)

            base_x = tip[0] - unit_x * head_length
            base_y = tip[1] - unit_y * head_length
            perpendicular_x = -unit_y * head_width
            perpendicular_y = unit_x * head_width
            draw.polygon(
                (
                    tip,
                    (base_x + perpendicular_x, base_y + perpendicular_y),
                    (base_x - perpendicular_x, base_y - perpendicular_y),
                ),
                fill=color,
            )

    return np.asarray(Image.alpha_composite(image, overlay))


def _sample_tile_field(
    source,
    field: WeatherField,
    timestamp: int,
    z: int,
    x: int,
    y: int,
    tile_size: int,
) -> np.ndarray:
    values = np.asarray(
        source.sample_tile_field(
            field,
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
    return values


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
    vector_style: str = "",
) -> bytes:
    """Sample one physical field tile, LUT-colorize it, and encode it."""

    normalized = WeatherField(field)
    if palette.field is not normalized:
        raise ValueError(
            f"palette {palette.id} does not support {normalized.value}"
        )
    if vector_style and normalized is not WeatherField.WIND_SPEED_10M:
        raise ValueError("wind vectors are only supported for wind_speed_10m")
    values = _sample_tile_field(
        source, normalized, timestamp, z, x, y, tile_size,
    )
    rgba = colorize_weather_values(values, palette)
    if vector_style:
        u_values = _sample_tile_field(
            source, WeatherField.WIND_U_10M, timestamp, z, x, y, tile_size,
        )
        v_values = _sample_tile_field(
            source, WeatherField.WIND_V_10M, timestamp, z, x, y, tile_size,
        )
        rgba = draw_wind_vectors(rgba, u_values, v_values, vector_style)
    return encode_weather_image(rgba, fmt)
