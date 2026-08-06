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
