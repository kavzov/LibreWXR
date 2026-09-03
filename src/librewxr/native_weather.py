# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Optional native kernels with a strict NumPy fallback.

The public functions in this module are the only integration boundary used by
the renderer. The separately-installed ``_librewxr_native`` module is never a
base dependency, so a normal Hatchling install needs neither Rust nor maturin.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from librewxr.config import settings
from librewxr.data.weather_fields import (
    relative_humidity_from_temperature_dewpoint,
    wind_speed_from_uv,
)
from librewxr.data.weather_sampling import SamplingPlan

try:
    import _librewxr_native as _native
except ImportError as exc:  # pragma: no cover - branch depends on local install
    _native = None
    _native_import_error: ImportError | None = exc
else:
    _native_import_error = None

Implementation = Literal["python", "rust"]


def native_available() -> bool:
    """Whether the optional PyO3 module is importable in this process."""

    return _native is not None


def ensure_native_render_available() -> None:
    """Enforce ``LIBREWXR_NATIVE_RENDER=on`` during source construction."""

    if settings.native_render == "on" and _native is None:
        detail = f": {_native_import_error}" if _native_import_error else ""
        raise RuntimeError(
            "LIBREWXR_NATIVE_RENDER=on, but the optional _librewxr_native "
            "extension is unavailable. Install a prebuilt librewxr-native "
            f"wheel or use auto/off{detail}"
        )


def active_implementation(
    implementation: Implementation | None = None,
) -> Implementation:
    """Resolve an explicit test override or the configured runtime backend."""

    if implementation == "python":
        return "python"
    if implementation == "rust":
        if _native is None:
            raise RuntimeError("Rust weather sampling requested but extension is unavailable")
        return "rust"
    if implementation is not None:
        raise ValueError(f"unknown weather sampling implementation: {implementation}")
    if settings.native_render == "off":
        return "python"
    if settings.native_render == "on":
        ensure_native_render_available()
        return "rust"
    return "rust" if _native is not None else "python"


def _validate_plan(plan: SamplingPlan) -> tuple[int, ...]:
    shape = plan.shape
    expected = {
        "r0": np.dtype(np.int32),
        "r1": np.dtype(np.int32),
        "c0": np.dtype(np.int32),
        "c1": np.dtype(np.int32),
        "dr": np.dtype(np.float32),
        "dc": np.dtype(np.float32),
        "valid": np.dtype(bool),
    }
    for name, dtype in expected.items():
        values = getattr(plan, name)
        if values.shape != shape:
            raise ValueError(
                f"sampling plan {name} shape {values.shape} does not match {shape}"
            )
        if values.dtype != dtype:
            raise TypeError(
                f"sampling plan {name} must have dtype {dtype}, got {values.dtype}"
            )
        if not values.flags.c_contiguous:
            raise ValueError(f"sampling plan {name} must be C-contiguous")
    return shape


def _validate_frame(frame: np.ndarray, name: str = "frame") -> np.ndarray:
    values = np.asarray(frame)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {values.shape}")
    if values.dtype not in (np.dtype(np.int16), np.dtype(np.uint16)):
        raise TypeError(f"{name} must have dtype int16 or uint16, got {values.dtype}")
    if not values.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return values


def _validate_float_pair(
    left: np.ndarray,
    right: np.ndarray,
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left)
    right = np.asarray(right)
    for values, name in ((left, left_name), (right, right_name)):
        if values.dtype != np.float32:
            raise TypeError(f"{name} must have dtype float32, got {values.dtype}")
        if not values.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    if left.shape != right.shape:
        raise ValueError(
            f"{right_name} shape {right.shape} does not match {left_name} {left.shape}"
        )
    return left, right


def _validate_bounds(frame: np.ndarray, plan: SamplingPlan) -> None:
    height, width = frame.shape
    if any(
        index.size
        and (int(index.min()) < 0 or int(index.max()) >= bound)
        for index, bound in (
            (plan.r0, height),
            (plan.r1, height),
            (plan.c0, width),
            (plan.c1, width),
        )
    ):
        raise IndexError(
            f"sampling plan contains indexes outside frame shape {frame.shape}"
        )
    if (
        not np.isfinite(plan.dr).all()
        or not np.isfinite(plan.dc).all()
        or np.any((plan.dr < 0.0) | (plan.dr > 1.0))
        or np.any((plan.dc < 0.0) | (plan.dc > 1.0))
    ):
        raise ValueError("sampling weights must be finite and within [0, 1]")


def _numpy_sample(
    frame: np.ndarray,
    plan: SamplingPlan,
    *,
    scale: float,
    offset: float,
    nodata: int | None,
    bilinear: bool,
) -> np.ndarray:
    _validate_bounds(frame, plan)
    if not bilinear:
        encoded = frame[plan.r0, plan.c0]
        result = encoded.astype(np.float32)
        result *= np.float32(scale)
        result += np.float32(offset)
        if nodata is not None:
            result[encoded == nodata] = np.nan
        result[~plan.valid] = np.nan
        return result

    samples = (
        frame[plan.r0, plan.c0],
        frame[plan.r0, plan.c1],
        frame[plan.r1, plan.c0],
        frame[plan.r1, plan.c1],
    )
    has_nodata = nodata is not None and any(
        np.any(sample == nodata) for sample in samples
    )
    if not has_nodata:
        top = samples[0].astype(np.float32)
        scratch = samples[1].astype(np.float32)
        scratch -= top
        scratch *= plan.dc
        top += scratch
        bottom = samples[2].astype(np.float32)
        scratch = samples[3].astype(np.float32)
        scratch -= bottom
        scratch *= plan.dc
        bottom += scratch
        bottom -= top
        bottom *= plan.dr
        top += bottom
        top *= np.float32(scale)
        top += np.float32(offset)
        top[~plan.valid] = np.nan
        return top

    weighted = np.zeros(plan.shape, dtype=np.float32)
    weight_sum = np.zeros(plan.shape, dtype=np.float32)
    scratch = np.empty(plan.shape, dtype=np.float32)
    one_minus_dr = np.subtract(np.float32(1.0), plan.dr)
    one_minus_dc = np.subtract(np.float32(1.0), plan.dc)
    weights = (
        one_minus_dr * one_minus_dc,
        one_minus_dr * plan.dc,
        plan.dr * one_minus_dc,
        plan.dr * plan.dc,
    )
    for sample, weight in zip(samples, weights, strict=True):
        valid = sample != nodata
        scratch.fill(0.0)
        np.multiply(sample, weight, out=scratch, where=valid)
        weighted += scratch
        np.add(weight_sum, weight, out=weight_sum, where=valid)
    result = np.full(plan.shape, np.nan, dtype=np.float32)
    valid_result = (weight_sum > 0.0) & plan.valid
    np.divide(weighted, weight_sum, out=result, where=valid_result)
    result *= np.float32(scale)
    result[valid_result] += np.float32(offset)
    return result


def _native_arguments(plan: SamplingPlan) -> tuple[np.ndarray, ...]:
    return (
        plan.r0.reshape(1, plan.r0.size),
        plan.r1.reshape(1, plan.r1.size),
        plan.c0.reshape(1, plan.c0.size),
        plan.c1.reshape(1, plan.c1.size),
        plan.dr.reshape(1, plan.dr.size),
        plan.dc.reshape(1, plan.dc.size),
        plan.valid.reshape(1, plan.valid.size),
    )


def sample_bilinear_regular_grid(
    frame: np.ndarray,
    plan: SamplingPlan,
    *,
    scale: float,
    offset: float,
    nodata: int | None,
    bilinear: bool = True,
    implementation: Implementation | None = None,
) -> np.ndarray:
    """Sample one encoded regular-grid frame and decode physical values."""

    frame = _validate_frame(frame)
    _validate_plan(plan)
    if active_implementation(implementation) == "python":
        return _numpy_sample(
            frame,
            plan,
            scale=scale,
            offset=offset,
            nodata=nodata,
            bilinear=bilinear,
        )
    function = _native.sample_i16 if frame.dtype == np.int16 else _native.sample_u16
    result = function(
        frame,
        *_native_arguments(plan),
        scale,
        offset,
        nodata,
        bilinear,
    )
    return result.reshape(plan.shape)


def sample_temporal_bilinear(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    plan: SamplingPlan,
    *,
    alpha: float,
    scale: float,
    offset: float,
    nodata: int | None,
    bilinear: bool = True,
    implementation: Implementation | None = None,
) -> np.ndarray:
    """Fuse spatial sampling, temporal interpolation, and affine decode."""

    frame_a = _validate_frame(frame_a, "frame_a")
    frame_b = _validate_frame(frame_b, "frame_b")
    if frame_a.shape != frame_b.shape or frame_a.dtype != frame_b.dtype:
        raise ValueError(
            "frame_b shape and dtype must match frame_a: "
            f"{frame_b.shape}/{frame_b.dtype} != {frame_a.shape}/{frame_a.dtype}"
        )
    _validate_plan(plan)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("temporal alpha must be finite and within [0, 1]")
    if active_implementation(implementation) == "python":
        before = _numpy_sample(
            frame_a,
            plan,
            scale=scale,
            offset=offset,
            nodata=nodata,
            bilinear=bilinear,
        )
        after = _numpy_sample(
            frame_b,
            plan,
            scale=scale,
            offset=offset,
            nodata=nodata,
            bilinear=bilinear,
        )
        valid_before = np.isfinite(before)
        valid_after = np.isfinite(after)
        both = valid_before & valid_after
        before[both] += np.float32(alpha) * (after[both] - before[both])
        before[~valid_before & valid_after] = after[~valid_before & valid_after]
        return before
    function = (
        _native.sample_temporal_i16
        if frame_a.dtype == np.int16
        else _native.sample_temporal_u16
    )
    result = function(
        frame_a,
        frame_b,
        *_native_arguments(plan),
        alpha,
        scale,
        offset,
        nodata,
        bilinear,
    )
    return result.reshape(plan.shape)


def sample_precipitation_regular_grid(
    frame: np.ndarray,
    plan: SamplingPlan,
    *,
    bilinear: bool = True,
    implementation: Implementation | None = None,
) -> np.ndarray:
    """Sample encoded precipitation with the legacy zero-aware semantics.

    Unlike generic continuous fields, an encoded zero means both the bottom of
    the reflectivity scale and a dry/missing neighbour. Historical LibreWXR
    rendering falls back to the north-west sample when any bilinear corner is
    zero; retaining that rule keeps the tile-aware path pixel-identical.
    """

    frame = np.asarray(frame)
    if frame.ndim != 2 or frame.dtype != np.uint8:
        raise TypeError("frame must be a two-dimensional uint8 array")
    if not frame.flags.c_contiguous:
        raise ValueError("frame must be C-contiguous")
    _validate_plan(plan)
    _validate_bounds(frame, plan)
    if active_implementation(implementation) == "rust":
        return _native.sample_precipitation_u8(
            frame,
            *_native_arguments(plan),
            bilinear,
        ).reshape(plan.shape)

    v00 = frame[plan.r0, plan.c0]
    if not bilinear:
        result = v00.copy()
        result[~plan.valid] = 0
        return result
    v01 = frame[plan.r0, plan.c1]
    v10 = frame[plan.r1, plan.c0]
    v11 = frame[plan.r1, plan.c1]
    any_zero = (v00 == 0) | (v01 == 0) | (v10 == 0) | (v11 == 0)
    row_weight = plan.dr
    col_weight = plan.dc
    interpolated = (
        v00.astype(np.float32) * (1.0 - row_weight) * (1.0 - col_weight)
        + v01.astype(np.float32) * (1.0 - row_weight) * col_weight
        + v10.astype(np.float32) * row_weight * (1.0 - col_weight)
        + v11.astype(np.float32) * row_weight * col_weight
    )
    result = np.where(any_zero, v00, interpolated + 0.5)
    result = np.clip(result, 0, 255).astype(np.uint8)
    result[~plan.valid] = 0
    return result


def sample_derived_humidity(
    temperature: np.ndarray,
    dewpoint: np.ndarray,
    *,
    implementation: Implementation | None = None,
) -> np.ndarray:
    """Calculate relative humidity with identical Python/Rust semantics."""

    temperature, dewpoint = _validate_float_pair(
        temperature, dewpoint, "temperature", "dewpoint"
    )
    if active_implementation(implementation) == "rust":
        return _native.sample_derived_humidity(
            temperature.reshape(1, temperature.size),
            dewpoint.reshape(1, dewpoint.size),
        ).reshape(temperature.shape)
    return relative_humidity_from_temperature_dewpoint(temperature, dewpoint)


def sample_wind_speed(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    *,
    implementation: Implementation | None = None,
) -> np.ndarray:
    """Calculate wind speed with identical Python/Rust semantics."""

    wind_u, wind_v = _validate_float_pair(wind_u, wind_v, "wind_u", "wind_v")
    if active_implementation(implementation) == "rust":
        return _native.sample_wind_speed(
            wind_u.reshape(1, wind_u.size),
            wind_v.reshape(1, wind_v.size),
        ).reshape(wind_u.shape)
    return wind_speed_from_uv(wind_u, wind_v)


def sample_radar_bilinear(
    frame: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    *,
    implementation: Implementation | None = None,
) -> np.ndarray:
    """Sample an encoded uint8 radar grid with zero-aware interpolation.

    A zero corner makes the result fall back to the nearest value. Coordinates
    outside the source grid produce zero; masked tile-coordinate plans use
    ``-1`` for those pixels. This keeps the native path pixel-identical to the
    historical NumPy renderer while avoiding a second integer coordinate pair
    solely for its out-of-bounds mask.
    """

    frame = np.asarray(frame)
    row = np.asarray(row)
    col = np.asarray(col)
    if frame.ndim != 2 or frame.dtype != np.uint8:
        raise TypeError("frame must be a two-dimensional uint8 array")
    if row.ndim != 2 or row.dtype != np.float32:
        raise TypeError("row must be a two-dimensional float32 array")
    if col.shape != row.shape or col.dtype != np.float32:
        raise TypeError("col must be a float32 array matching row")
    for name, array in (("frame", frame), ("row", row), ("col", col)):
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    if active_implementation(implementation) == "rust":
        return _native.sample_radar_bilinear_u8(frame, row, col)

    if not np.isfinite(row).all() or not np.isfinite(col).all():
        raise ValueError("row and col coordinates must be finite")
    valid = (
        (row >= 0)
        & (col >= 0)
        & (row <= frame.shape[0] - 1)
        & (col <= frame.shape[1] - 1)
    )
    safe_row = np.clip(row, 0, frame.shape[0] - 1)
    safe_col = np.clip(col, 0, frame.shape[1] - 1)
    r0 = np.floor(safe_row).astype(np.int32)
    c0 = np.floor(safe_col).astype(np.int32)
    r1 = np.minimum(r0 + 1, frame.shape[0] - 1)
    c1 = np.minimum(c0 + 1, frame.shape[1] - 1)
    dr = (safe_row - r0).astype(np.float32)
    dc = (safe_col - c0).astype(np.float32)
    v00 = frame[r0, c0].astype(np.float32)
    v01 = frame[r0, c1].astype(np.float32)
    v10 = frame[r1, c0].astype(np.float32)
    v11 = frame[r1, c1].astype(np.float32)
    any_zero = (v00 == 0) | (v01 == 0) | (v10 == 0) | (v11 == 0)
    interpolated = (
        v00 * (1 - dr) * (1 - dc)
        + v01 * (1 - dr) * dc
        + v10 * dr * (1 - dc)
        + v11 * dr * dc
    )
    result = np.clip(
        np.where(any_zero, v00, interpolated) + 0.5, 0, 255,
    ).astype(np.uint8)
    result[~valid] = 0
    return result


def blend_radar_nowcast(
    radar: np.ndarray,
    model: np.ndarray,
    model_raw: np.ndarray,
    feather: np.ndarray,
    blend_weight: float,
    pixel_threshold: int | None,
    *,
    implementation: Implementation | None = None,
) -> np.ndarray:
    """Blend radar/model fields in one allocation-free native pixel kernel."""

    radar = np.asarray(radar)
    model = np.asarray(model)
    model_raw = np.asarray(model_raw)
    feather = np.asarray(feather)
    shape = radar.shape
    expected = (
        ("radar", radar, np.dtype(np.uint8)),
        ("model", model, np.dtype(np.float32)),
        ("model_raw", model_raw, np.dtype(np.uint8)),
        ("feather", feather, np.dtype(np.float32)),
    )
    for name, array, dtype in expected:
        if array.ndim != 2 or array.shape != shape or array.dtype != dtype:
            raise TypeError(f"{name} must be a {dtype} array matching radar")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(blend_weight) or not 0.0 <= blend_weight <= 1.0:
        raise ValueError("blend_weight must be finite and within [0, 1]")
    if pixel_threshold is not None and not 0 <= pixel_threshold <= 255:
        raise ValueError("pixel_threshold must be within [0, 255]")

    if active_implementation(implementation) == "rust":
        return _native.blend_radar_nowcast_u8(
            radar, model, model_raw, feather, blend_weight, pixel_threshold,
        )

    adjusted_model = model
    if blend_weight > 0 and pixel_threshold is not None:
        dry_model = model < pixel_threshold
        live_radar = radar >= pixel_threshold
        adjusted_model = np.where(
            dry_model & live_radar, pixel_threshold, model,
        )
    effective_weight = blend_weight * feather
    blended = (
        effective_weight * radar.astype(np.float32)
        + (1.0 - effective_weight) * adjusted_model
    )
    result = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    result[(radar == 0) & (model_raw == 0)] = 0
    return result


def colorize_radar(
    values: np.ndarray,
    rain_lut: np.ndarray,
    *,
    snow_lut: np.ndarray | None = None,
    snow_mask: np.ndarray | None = None,
    display_threshold: int | None = None,
    implementation: Implementation | None = None,
) -> np.ndarray:
    """Apply radar LUT, phase mask, and display threshold in one kernel."""

    values = np.asarray(values)
    rain_lut = np.asarray(rain_lut)
    if values.ndim != 2 or values.dtype != np.uint8:
        raise TypeError("values must be a two-dimensional uint8 array")
    if rain_lut.shape != (256, 4) or rain_lut.dtype != np.uint8:
        raise TypeError("rain_lut must be a uint8 array with shape (256, 4)")
    if (snow_lut is None) != (snow_mask is None):
        raise ValueError("snow_lut and snow_mask must be provided together")
    if snow_lut is not None:
        snow_lut = np.asarray(snow_lut)
        snow_mask = np.asarray(snow_mask)
        if snow_lut.shape != (256, 4) or snow_lut.dtype != np.uint8:
            raise TypeError("snow_lut must be a uint8 array with shape (256, 4)")
        if snow_mask.shape != values.shape or snow_mask.dtype != np.bool_:
            raise TypeError("snow_mask must be a bool array matching values")
    if display_threshold is not None and not 0 <= display_threshold <= 255:
        raise ValueError("display_threshold must be within [0, 255]")
    arrays = [("values", values), ("rain_lut", rain_lut)]
    if snow_lut is not None:
        arrays.extend((("snow_lut", snow_lut), ("snow_mask", snow_mask)))
    for name, array in arrays:
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")

    if active_implementation(implementation) == "rust":
        return _native.colorize_radar_u8(
            values, rain_lut, snow_lut, snow_mask, display_threshold
        )

    display_values = values
    if display_threshold is not None:
        display_values = values.copy()
        display_values[display_values < display_threshold] = 0
    rain = rain_lut[display_values]
    if snow_mask is None:
        return rain
    snow = snow_lut[display_values]
    return np.where(snow_mask[..., np.newaxis], snow, rain)


def encode_radar_png(
    rgba: np.ndarray,
    *,
    implementation: Implementation | None = None,
) -> bytes | None:
    """Encode RGBA through Rust, or return ``None`` for the Pillow fallback."""

    if active_implementation(implementation) == "python":
        return None
    rgba = np.asarray(rgba)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise TypeError("rgba must be a uint8 array with shape (height, width, 4)")
    if not rgba.flags.c_contiguous:
        rgba = np.ascontiguousarray(rgba)
    return bytes(_native.encode_png_rgba(rgba))
