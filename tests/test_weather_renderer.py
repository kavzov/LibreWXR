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
from librewxr.tiles.weather_renderer import (
    colorize_weather_values,
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
