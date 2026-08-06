# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for scalar-weather palettes and rendering."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from librewxr.colors.weather_palettes import (
    WEATHER_LUT_SIZE,
    WEATHER_PALETTES,
)
from librewxr.data.weather_fields import WeatherField
from librewxr.config import settings
from librewxr.tiles.png_palette import encode_png
from librewxr.tiles.weather_renderer import (
    colorize_weather_values,
    encode_weather_image,
    render_scalar_weather_tile,
)


def test_weather_palette_registry_has_stable_prebuilt_luts():
    assert list(WEATHER_PALETTES) == [
        "temperature",
        "dewpoint",
        "humidity",
        "pressure",
        "wind_speed",
    ]
    for palette in WEATHER_PALETTES.values():
        assert palette.lut.shape == (WEATHER_LUT_SIZE, 4)
        assert palette.lut.dtype == np.uint8
        assert not palette.lut.flags.writeable
        assert palette.stops[0].value <= palette.minimum
        assert palette.stops[-1].value >= palette.maximum


def test_colorize_uses_below_above_nodata_and_palette_stops():
    palette = WEATHER_PALETTES["humidity"]
    values = np.array([[-1.0, 0.0, 60.0, 100.0, 101.0, np.nan]])

    rgba = colorize_weather_values(values, palette)

    np.testing.assert_array_equal(rgba[0, 0], palette.below_rgba)
    np.testing.assert_array_equal(rgba[0, 1], palette.lut[0])
    np.testing.assert_allclose(
        rgba[0, 2],
        [184, 224, 210, 255],
        atol=1,
    )
    np.testing.assert_array_equal(rgba[0, 3], palette.lut[-1])
    np.testing.assert_array_equal(rgba[0, 4], palette.above_rgba)
    np.testing.assert_array_equal(rgba[0, 5], palette.nodata_rgba)


def test_optimized_colorize_matches_reference_pixels_exactly():
    palette = WEATHER_PALETTES["temperature"]
    rng = np.random.default_rng(20260806)
    values = rng.uniform(-70.0, 70.0, (128, 128)).astype(np.float32)
    values.flat[::257] = np.nan

    physical = np.asarray(values, dtype=np.float32)
    expected = np.empty((*physical.shape, 4), dtype=np.uint8)
    finite = np.isfinite(physical)
    expected[~finite] = palette.nodata_rgba
    below = finite & (physical < palette.minimum)
    above = finite & (physical > palette.maximum)
    inside = finite & ~below & ~above
    expected[below] = palette.below_rgba
    expected[above] = palette.above_rgba
    normalized = (
        (physical[inside] - palette.minimum)
        / (palette.maximum - palette.minimum)
    )
    indices = np.rint(normalized * (len(palette.lut) - 1)).astype(np.int32)
    expected[inside] = palette.lut[indices]

    np.testing.assert_array_equal(
        colorize_weather_values(values, palette),
        expected,
    )


def test_exact_png8_path_is_unchanged_when_quantization_is_enabled():
    rgba = np.zeros((40, 64, 4), dtype=np.uint8)
    for index in range(40):
        rgba[index] = (index * 5, index * 3, index * 2, 255 - index)
    image = Image.fromarray(rgba, "RGBA")

    assert encode_png(image, quantize=True) == encode_png(image)


def test_quantized_weather_png_is_smaller_deterministic_and_within_tolerance(
    monkeypatch,
):
    axis = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    x_coord, y_coord = np.meshgrid(axis, axis)
    values = (
        12.0
        + 24.0 * np.sin(y_coord * 1.7)
        + 8.0 * np.cos(x_coord * 3.1)
        + 2.0 * np.sin((x_coord + y_coord) * 9.0)
    )
    values[:12, :12] = np.nan
    rgba = colorize_weather_values(values, WEATHER_PALETTES["temperature"])

    monkeypatch.setattr(settings, "weather_png_mode", "lossless")
    lossless = encode_weather_image(rgba, "png")
    lossless_rgba = np.asarray(
        Image.open(io.BytesIO(lossless)).convert("RGBA")
    )
    np.testing.assert_array_equal(lossless_rgba, rgba)

    monkeypatch.setattr(settings, "weather_png_mode", "quantized")
    monkeypatch.setattr(settings, "weather_png_colors", 256)
    monkeypatch.setattr(settings, "weather_png_dither", False)
    quantized = encode_weather_image(rgba, "png")
    repeated = encode_weather_image(rgba, "png")
    quantized_image = Image.open(io.BytesIO(quantized))
    quantized_rgba = np.asarray(quantized_image.convert("RGBA"))

    assert quantized == repeated
    assert quantized_image.mode == "P"
    assert len(quantized) < len(lossless) * 0.5
    assert np.array_equal(quantized_rgba[..., 3], rgba[..., 3])
    rgb_error = np.abs(
        quantized_rgba[..., :3].astype(np.int16)
        - rgba[..., :3].astype(np.int16)
    )
    assert rgb_error.mean() < 3.0
    assert np.percentile(rgb_error, 95) <= 10.0
    assert rgb_error.max() <= 20


class _ConstantTileSource:
    def __init__(self, value: float):
        self.value = value
        self.calls = []

    def sample_tile_field(self, field, z, x, y, **kwargs):
        self.calls.append((field, z, x, y, kwargs))
        size = kwargs["tile_size"]
        return np.full((size, size), self.value, dtype=np.float32)


def test_renderer_samples_physical_values_without_smoothing():
    source = _ConstantTileSource(20.0)
    rendered = render_scalar_weather_tile(
        source=source,
        field=WeatherField.TEMPERATURE_2M,
        palette=WEATHER_PALETTES["temperature"],
        timestamp=1_700_000_000,
        z=2,
        x=1,
        y=1,
        tile_size=32,
        fmt="png",
    )

    image = Image.open(io.BytesIO(rendered)).convert("RGBA")
    assert image.size == (32, 32)
    assert source.calls == [
        (
            WeatherField.TEMPERATURE_2M,
            2,
            1,
            1,
            {
                "timestamp": 1_700_000_000,
                "tile_size": 32,
                "padding": 0,
                "bilinear": True,
            },
        )
    ]
    pixels = np.asarray(image)
    assert (pixels[..., 3] == 255).all()
    assert np.unique(pixels.reshape(-1, 4), axis=0).shape[0] == 1


def test_renderer_webp_signature():
    rendered = render_scalar_weather_tile(
        source=_ConstantTileSource(5.0),
        field=WeatherField.WIND_SPEED_10M,
        palette=WEATHER_PALETTES["wind_speed"],
        timestamp=1,
        z=0,
        x=0,
        y=0,
        tile_size=16,
        fmt="webp",
    )

    assert rendered[:4] == b"RIFF"
    assert rendered[8:12] == b"WEBP"
