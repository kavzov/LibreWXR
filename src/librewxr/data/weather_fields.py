# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Generic weather-field metadata, codecs, and derived-field calculations.

The field registry is the single source of truth for public identifiers,
physical units, on-disk encodings, valid ranges, and interpolation rules.
Source implementations expose samples in physical units through
``sample_field``; storage-specific scale/offset details stay here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np


class WeatherField(str, Enum):
    """Stable public identifiers for model-backed weather fields."""

    PRECIPITATION = "precipitation"
    TEMPERATURE_2M = "temperature_2m"
    DEWPOINT_2M = "dew_point_2m"
    RELATIVE_HUMIDITY_2M = "relative_humidity_2m"
    PRESSURE_MSL = "pressure_msl"
    WIND_U_10M = "wind_u_10m"
    WIND_V_10M = "wind_v_10m"
    WIND_SPEED_10M = "wind_speed_10m"


class SpatialInterpolation(str, Enum):
    """How a field may be sampled between native grid cells."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"


class TemporalInterpolation(str, Enum):
    """How a field may be sampled between native valid times."""

    NEAREST = "nearest"
    LINEAR = "linear"
    DERIVED = "derived"


@dataclass(frozen=True)
class FieldSpec:
    """Immutable storage and interpolation contract for one weather field.

    ``scale`` and ``offset`` use the conventional affine relationship
    ``physical = encoded * scale + offset``. ``nodata`` is an encoded-domain
    sentinel; decoded missing values are represented by ``NaN``.
    """

    field: WeatherField
    public_name: str
    unit: str
    storage_dtype: np.dtype
    scale: float
    offset: float
    nodata: int | None
    valid_range: tuple[float, float]
    spatial_interpolation: SpatialInterpolation
    temporal_interpolation: TemporalInterpolation
    derived: bool = False
    dependencies: tuple[WeatherField, ...] = ()

    @property
    def categorical(self) -> bool:
        """Whether values must use hard first-source selection."""

        return (
            self.spatial_interpolation is SpatialInterpolation.NEAREST
            and self.temporal_interpolation is TemporalInterpolation.NEAREST
        )


_FIELD_SPECS: dict[WeatherField, FieldSpec] = {
    WeatherField.PRECIPITATION: FieldSpec(
        field=WeatherField.PRECIPITATION,
        public_name="Precipitation reflectivity",
        unit="dBZ",
        storage_dtype=np.dtype("uint8"),
        scale=0.5,
        offset=-32.0,
        nodata=None,
        valid_range=(-32.0, 95.5),
        spatial_interpolation=SpatialInterpolation.BILINEAR,
        temporal_interpolation=TemporalInterpolation.LINEAR,
    ),
    WeatherField.TEMPERATURE_2M: FieldSpec(
        field=WeatherField.TEMPERATURE_2M,
        public_name="Temperature at 2 m",
        unit="°C",
        storage_dtype=np.dtype("int16"),
        scale=0.1,
        offset=0.0,
        nodata=-32768,
        valid_range=(-100.0, 60.0),
        spatial_interpolation=SpatialInterpolation.BILINEAR,
        temporal_interpolation=TemporalInterpolation.LINEAR,
    ),
    WeatherField.DEWPOINT_2M: FieldSpec(
        field=WeatherField.DEWPOINT_2M,
        public_name="Dew point at 2 m",
        unit="°C",
        storage_dtype=np.dtype("int16"),
        scale=0.1,
        offset=0.0,
        nodata=-32768,
        valid_range=(-120.0, 50.0),
        spatial_interpolation=SpatialInterpolation.BILINEAR,
        temporal_interpolation=TemporalInterpolation.LINEAR,
    ),
    WeatherField.RELATIVE_HUMIDITY_2M: FieldSpec(
        field=WeatherField.RELATIVE_HUMIDITY_2M,
        public_name="Relative humidity at 2 m",
        unit="%",
        storage_dtype=np.dtype("uint8"),
        scale=1.0,
        offset=0.0,
        nodata=255,
        valid_range=(0.0, 100.0),
        spatial_interpolation=SpatialInterpolation.BILINEAR,
        temporal_interpolation=TemporalInterpolation.DERIVED,
        derived=True,
        dependencies=(WeatherField.TEMPERATURE_2M, WeatherField.DEWPOINT_2M),
    ),
    WeatherField.PRESSURE_MSL: FieldSpec(
        field=WeatherField.PRESSURE_MSL,
        public_name="Mean sea-level pressure",
        unit="hPa",
        storage_dtype=np.dtype("uint16"),
        scale=0.1,
        offset=0.0,
        nodata=65535,
        valid_range=(800.0, 1100.0),
        spatial_interpolation=SpatialInterpolation.BILINEAR,
        temporal_interpolation=TemporalInterpolation.LINEAR,
    ),
    WeatherField.WIND_U_10M: FieldSpec(
        field=WeatherField.WIND_U_10M,
        public_name="Eastward wind at 10 m",
        unit="m/s",
        storage_dtype=np.dtype("int16"),
        scale=0.1,
        offset=0.0,
        nodata=-32768,
        valid_range=(-150.0, 150.0),
        spatial_interpolation=SpatialInterpolation.BILINEAR,
        temporal_interpolation=TemporalInterpolation.LINEAR,
    ),
    WeatherField.WIND_V_10M: FieldSpec(
        field=WeatherField.WIND_V_10M,
        public_name="Northward wind at 10 m",
        unit="m/s",
        storage_dtype=np.dtype("int16"),
        scale=0.1,
        offset=0.0,
        nodata=-32768,
        valid_range=(-150.0, 150.0),
        spatial_interpolation=SpatialInterpolation.BILINEAR,
        temporal_interpolation=TemporalInterpolation.LINEAR,
    ),
    WeatherField.WIND_SPEED_10M: FieldSpec(
        field=WeatherField.WIND_SPEED_10M,
        public_name="Wind speed at 10 m",
        unit="m/s",
        storage_dtype=np.dtype("uint16"),
        scale=0.1,
        offset=0.0,
        nodata=65535,
        valid_range=(0.0, 150.0),
        spatial_interpolation=SpatialInterpolation.BILINEAR,
        temporal_interpolation=TemporalInterpolation.DERIVED,
        derived=True,
        dependencies=(WeatherField.WIND_U_10M, WeatherField.WIND_V_10M),
    ),
}

FIELD_SPECS: Mapping[WeatherField, FieldSpec] = MappingProxyType(_FIELD_SPECS)


def field_spec(field: WeatherField) -> FieldSpec:
    """Return the immutable specification for ``field``."""

    try:
        return FIELD_SPECS[WeatherField(field)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unknown weather field: {field!r}") from exc


def encode_field(
    field: WeatherField,
    values: np.ndarray | float,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Encode physical values according to the central field registry.

    Finite values outside ``valid_range`` are clipped by default. Pass
    ``clip=False`` to make them an error. Non-finite values map to the encoded
    ``nodata`` sentinel; fields without a sentinel reject non-finite input.
    """

    spec = field_spec(field)
    physical = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(physical)
    lo, hi = spec.valid_range

    if not clip and np.any(finite & ((physical < lo) | (physical > hi))):
        raise ValueError(
            f"{spec.field.value} values must be within [{lo}, {hi}] {spec.unit}"
        )
    if spec.nodata is None and not finite.all():
        raise ValueError(f"{spec.field.value} has no nodata representation")

    bounded = np.clip(physical, lo, hi)
    encoded_float = np.rint((bounded - spec.offset) / spec.scale)
    dtype_info = np.iinfo(spec.storage_dtype)
    encoded_float = np.clip(encoded_float, dtype_info.min, dtype_info.max)

    if spec.nodata is not None:
        encoded_float = np.where(finite, encoded_float, spec.nodata)
    return encoded_float.astype(spec.storage_dtype)


def decode_field(field: WeatherField, values: np.ndarray | int) -> np.ndarray:
    """Decode stored values to float32 physical units and ``NaN`` nodata."""

    spec = field_spec(field)
    encoded = np.asarray(values)
    physical = encoded.astype(np.float32) * np.float32(spec.scale) + np.float32(
        spec.offset
    )
    if spec.nodata is not None:
        physical = np.where(encoded == spec.nodata, np.nan, physical).astype(
            np.float32,
            copy=False,
        )
    return physical


def relative_humidity_from_temperature_dewpoint(
    temperature_c: np.ndarray | float,
    dewpoint_c: np.ndarray | float,
) -> np.ndarray:
    """Calculate relative humidity with the Magnus saturation-vapour formula.

    Inputs and output broadcast like NumPy arrays. Missing/non-finite inputs
    remain ``NaN``; finite results are constrained to the physical 0–100% range.
    """

    temperature = np.asarray(temperature_c, dtype=np.float32)
    dewpoint = np.asarray(dewpoint_c, dtype=np.float32)
    temperature, dewpoint = np.broadcast_arrays(temperature, dewpoint)
    valid = np.isfinite(temperature) & np.isfinite(dewpoint)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        exponent = np.array(dewpoint, dtype=np.float32, copy=True)
        exponent *= np.float32(17.625)
        exponent /= np.float32(243.04) + dewpoint
        temperature_term = np.array(temperature, dtype=np.float32, copy=True)
        temperature_term *= np.float32(17.625)
        temperature_term /= np.float32(243.04) + temperature
        exponent -= temperature_term
        np.exp(exponent, out=exponent)
        exponent *= np.float32(100.0)
    np.clip(exponent, 0.0, 100.0, out=exponent)
    np.copyto(exponent, np.float32(np.nan), where=~valid)
    return exponent.astype(np.float32, copy=False)


def wind_speed_from_uv(
    wind_u: np.ndarray | float,
    wind_v: np.ndarray | float,
) -> np.ndarray:
    """Return wind speed in the same units as eastward/northward components."""

    u = np.asarray(wind_u, dtype=np.float32)
    v = np.asarray(wind_v, dtype=np.float32)
    u, v = np.broadcast_arrays(u, v)
    speed = np.asarray(np.hypot(u, v), dtype=np.float32)
    np.copyto(speed, np.float32(np.nan), where=~np.isfinite(speed))
    return speed.astype(np.float32, copy=False)


def wind_direction_from_uv(
    wind_u: np.ndarray | float,
    wind_v: np.ndarray | float,
) -> np.ndarray:
    """Return meteorological wind-from direction in degrees clockwise north."""

    u = np.asarray(wind_u, dtype=np.float64)
    v = np.asarray(wind_v, dtype=np.float64)
    u, v = np.broadcast_arrays(u, v)
    speed = np.hypot(u, v)
    valid = np.isfinite(u) & np.isfinite(v) & (speed > 0.0)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    direction = np.where(valid, direction, np.nan)
    return direction.astype(np.float32)


class WeatherFieldSourceMixin:
    """Backward-compatible generic-field surface for precipitation sources.

    Existing regional grids inherit this mixin and therefore need no new model
    variables. Future sources can override these methods as they add physical
    fields; the legacy precipitation path continues to delegate to ``sample``.
    """

    def available_fields(self) -> frozenset[WeatherField]:
        return frozenset({WeatherField.PRECIPITATION})

    def has_field(self, field: WeatherField) -> bool:
        try:
            normalized = WeatherField(field)
        except ValueError:
            return False
        return normalized in self.available_fields()

    def sample_field(
        self,
        field: WeatherField,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = True,
    ) -> np.ndarray:
        normalized = WeatherField(field)
        if normalized is not WeatherField.PRECIPITATION:
            raise KeyError(f"{self.name} does not provide {normalized.value}")
        encoded = self.sample(lat, lon, timestamp, bilinear)
        return decode_field(WeatherField.PRECIPITATION, encoded)
