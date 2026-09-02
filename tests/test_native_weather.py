# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Parity and validation tests for optional PyO3 weather kernels."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from librewxr.config import settings
from librewxr.data.weather_fields import WeatherField, field_spec
from librewxr.data.weather_sampling import build_regular_sampling_plan
from librewxr import native_weather
from librewxr.native_weather import (
    active_implementation,
    ensure_native_render_available,
    colorize_radar,
    encode_radar_png,
    sample_radar_bilinear,
    sample_bilinear_regular_grid,
    sample_derived_humidity,
    sample_temporal_bilinear,
    sample_wind_speed,
)

pytestmark = pytest.mark.ecmwf


def _plan(lat: np.ndarray, lon: np.ndarray):
    return build_regular_sampling_plan(
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
        west=-180.0,
        north=90.0,
        pixel_size_x=5.0,
        pixel_size_y=5.0,
        width=72,
        height=37,
        wrap_longitude=True,
    )


def _random_frames(dtype: np.dtype, seed: int = 20260806):
    rng = np.random.default_rng(seed)
    if dtype == np.int16:
        first = rng.integers(-900, 500, (37, 72), dtype=np.int16)
        second = rng.integers(-900, 500, (37, 72), dtype=np.int16)
        nodata = -32768
    else:
        first = rng.integers(8000, 11000, (37, 72), dtype=np.uint16)
        second = rng.integers(8000, 11000, (37, 72), dtype=np.uint16)
        nodata = 65535
    first[3, 4] = nodata
    second[12, 30] = nodata
    return first, second, nodata


def _assert_parity(python: np.ndarray, rust: np.ndarray, scale: float):
    assert rust.dtype == python.dtype == np.float32
    assert rust.shape == python.shape
    np.testing.assert_allclose(
        rust,
        python,
        atol=scale / 2.0 + 1e-6,
        rtol=0.0,
        equal_nan=True,
    )


@pytest.fixture
def require_native():
    if not native_weather.native_available():
        pytest.skip("optional _librewxr_native extension is not installed")


@pytest.mark.parametrize(
    ("dtype", "field"),
    [
        (np.dtype(np.int16), WeatherField.TEMPERATURE_2M),
        (np.dtype(np.uint16), WeatherField.PRESSURE_MSL),
    ],
)
@pytest.mark.parametrize("bilinear", [False, True])
def test_native_nearest_and_bilinear_match_numpy(
    require_native,
    dtype,
    field,
    bilinear,
):
    frame, _second, nodata = _random_frames(dtype)
    latitude = np.linspace(84.0, -84.0, 31, dtype=np.float64)[:, None]
    longitude = np.linspace(-179.0, 179.0, 43, dtype=np.float64)[None, :]
    plan = _plan(
        np.broadcast_to(latitude, (31, 43)),
        np.broadcast_to(longitude, (31, 43)),
    )
    spec = field_spec(field)
    kwargs = dict(
        scale=spec.scale,
        offset=spec.offset,
        nodata=nodata,
        bilinear=bilinear,
    )

    python = sample_bilinear_regular_grid(
        frame, plan, implementation="python", **kwargs
    )
    rust = sample_bilinear_regular_grid(
        frame, plan, implementation="rust", **kwargs
    )

    _assert_parity(python, rust, spec.scale)


@pytest.mark.parametrize(
    ("dtype", "field"),
    [
        (np.dtype(np.int16), WeatherField.DEWPOINT_2M),
        (np.dtype(np.uint16), WeatherField.PRESSURE_MSL),
    ],
)
@pytest.mark.parametrize("bilinear", [False, True])
def test_native_fused_temporal_matches_numpy(
    require_native,
    dtype,
    field,
    bilinear,
):
    first, second, nodata = _random_frames(dtype)
    rng = np.random.default_rng(4815162342)
    lat = rng.uniform(-85.0, 85.0, (27, 35))
    lon = rng.uniform(-540.0, 540.0, (27, 35))
    plan = _plan(lat, lon)
    spec = field_spec(field)
    kwargs = dict(
        alpha=0.37,
        scale=spec.scale,
        offset=spec.offset,
        nodata=nodata,
        bilinear=bilinear,
    )

    python = sample_temporal_bilinear(
        first, second, plan, implementation="python", **kwargs
    )
    rust = sample_temporal_bilinear(
        first, second, plan, implementation="rust", **kwargs
    )

    _assert_parity(python, rust, spec.scale)


def test_native_dateline_grid_boundaries_and_nodata(require_native):
    frame, second, nodata = _random_frames(np.dtype(np.int16))
    frame[0, 0] = nodata
    second[-1, -1] = nodata
    lat = np.array([[0.0, 0.0, 0.0, 90.0, -90.0]])
    lon = np.array([[-180.0, 180.0, 540.0, -180.001, 179.999]])
    plan = _plan(lat, lon)
    spec = field_spec(WeatherField.TEMPERATURE_2M)
    kwargs = dict(
        alpha=0.5,
        scale=spec.scale,
        offset=spec.offset,
        nodata=nodata,
        bilinear=True,
    )

    python = sample_temporal_bilinear(
        frame, second, plan, implementation="python", **kwargs
    )
    rust = sample_temporal_bilinear(
        frame, second, plan, implementation="rust", **kwargs
    )

    _assert_parity(python, rust, spec.scale)
    np.testing.assert_allclose(rust[0, 0], rust[0, 1], atol=spec.scale / 2)


def test_native_humidity_and_wind_speed_match_numpy(require_native):
    rng = np.random.default_rng(8675309)
    temperature = np.ascontiguousarray(
        rng.uniform(-80.0, 50.0, (67, 59)).astype(np.float32)
    )
    dewpoint = np.ascontiguousarray(
        np.minimum(temperature, rng.uniform(-100.0, 45.0, temperature.shape))
        .astype(np.float32)
    )
    wind_u = np.ascontiguousarray(
        rng.uniform(-80.0, 80.0, temperature.shape).astype(np.float32)
    )
    wind_v = np.ascontiguousarray(
        rng.uniform(-80.0, 80.0, temperature.shape).astype(np.float32)
    )
    temperature[0, 0] = np.nan
    wind_v[1, 1] = np.nan

    humidity_python = sample_derived_humidity(
        temperature, dewpoint, implementation="python"
    )
    humidity_rust = sample_derived_humidity(
        temperature, dewpoint, implementation="rust"
    )
    wind_python = sample_wind_speed(wind_u, wind_v, implementation="python")
    wind_rust = sample_wind_speed(wind_u, wind_v, implementation="rust")

    _assert_parity(humidity_python, humidity_rust, 1.0)
    _assert_parity(wind_python, wind_rust, 0.1)


def test_native_radar_bilinear_matches_numpy(require_native):
    rng = np.random.default_rng(314159)
    frame = rng.integers(0, 256, (73, 91), dtype=np.uint8)
    frame[frame < 48] = 0
    row = np.ascontiguousarray(
        rng.uniform(0, frame.shape[0] - 1, (37, 43)).astype(np.float32)
    )
    col = np.ascontiguousarray(
        rng.uniform(0, frame.shape[1] - 1, (37, 43)).astype(np.float32)
    )

    python = sample_radar_bilinear(frame, row, col, implementation="python")
    rust = sample_radar_bilinear(frame, row, col, implementation="rust")

    np.testing.assert_array_equal(rust, python)


def test_native_radar_colorize_and_png_are_lossless(require_native):
    rng = np.random.default_rng(271828)
    values = rng.integers(0, 256, (47, 59), dtype=np.uint8)
    snow_mask = np.ascontiguousarray(rng.random(values.shape) > 0.7)
    rain_lut = rng.integers(0, 256, (256, 4), dtype=np.uint8)
    snow_lut = rng.integers(0, 256, (256, 4), dtype=np.uint8)
    kwargs = {
        "snow_lut": snow_lut,
        "snow_mask": snow_mask,
        "display_threshold": 108,
    }

    python = colorize_radar(
        values, rain_lut, implementation="python", **kwargs
    )
    rust = colorize_radar(values, rain_lut, implementation="rust", **kwargs)
    np.testing.assert_array_equal(rust, python)

    encoded = encode_radar_png(rust, implementation="rust")
    decoded = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGBA"))
    np.testing.assert_array_equal(decoded, rust)


def test_native_rejects_wrong_dtype_noncontiguous_shape_and_bounds(require_native):
    frame, _second, nodata = _random_frames(np.dtype(np.int16))
    plan = _plan(np.zeros((3, 4)), np.zeros((3, 4)))
    kwargs = dict(scale=0.1, offset=0.0, nodata=nodata, implementation="rust")

    with pytest.raises(TypeError, match="dtype"):
        sample_bilinear_regular_grid(frame.astype(np.float32), plan, **kwargs)
    with pytest.raises(ValueError, match="C-contiguous"):
        sample_bilinear_regular_grid(frame[:, ::-1], plan, **kwargs)

    bad_plan = _plan(np.zeros((3, 4)), np.zeros((3, 4)))
    bad_plan.r0.flags.writeable = True
    bad_plan.r0[0, 0] = 99
    bad_plan.r0.flags.writeable = False
    with pytest.raises(IndexError, match="outside frame shape"):
        sample_bilinear_regular_grid(frame, bad_plan, **kwargs)

    with pytest.raises(ValueError, match="does not match"):
        sample_derived_humidity(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((3, 2), dtype=np.float32),
            implementation="rust",
        )


def test_native_render_configuration_auto_on_off(monkeypatch):
    monkeypatch.setattr(settings, "native_render", "off")
    assert active_implementation() == "python"

    monkeypatch.setattr(settings, "native_render", "auto")
    expected = "rust" if native_weather.native_available() else "python"
    assert active_implementation() == expected

    monkeypatch.setattr(settings, "native_render", "on")
    if native_weather.native_available():
        assert active_implementation() == "rust"
    else:
        with pytest.raises(RuntimeError, match="extension is unavailable"):
            ensure_native_render_available()


def test_native_render_on_fails_cleanly_when_extension_missing(monkeypatch):
    monkeypatch.setattr(settings, "native_render", "on")
    monkeypatch.setattr(native_weather, "_native", None)
    monkeypatch.setattr(native_weather, "_native_import_error", ImportError("missing"))

    with pytest.raises(RuntimeError, match="LIBREWXR_NATIVE_RENDER=on"):
        ensure_native_render_available()


def test_auto_mode_uses_numpy_when_extension_is_missing(monkeypatch):
    frame, _second, nodata = _random_frames(np.dtype(np.int16))
    plan = _plan(np.zeros((2, 3)), np.zeros((2, 3)))
    monkeypatch.setattr(settings, "native_render", "auto")
    monkeypatch.setattr(native_weather, "_native", None)
    monkeypatch.setattr(native_weather, "_native_import_error", ImportError("missing"))

    result = sample_bilinear_regular_grid(
        frame,
        plan,
        scale=0.1,
        offset=0.0,
        nodata=nodata,
    )

    assert active_implementation() == "python"
    assert result.shape == (2, 3)
    assert np.isfinite(result).all()
