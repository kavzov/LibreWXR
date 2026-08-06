# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NWPSource Protocol and NWPChain dispatcher for multi-model NWP fallback.

Phase 1 of the multi-model NWP integration: defines the contract that any
numerical-weather-prediction source (ECMWF IFS, NOAA HRRR, DWD ICON-D2, ...)
must satisfy, plus a chain dispatcher that walks sources in priority order
and fills pixels from the first source with both coverage and data.

Each source handles its own quirks internally — Z-R conversion, projection
sampling, fetch cadence — so the renderer talks to a single uniform interface.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from librewxr.data.weather_fields import (
    WeatherField,
    decode_field,
    field_spec,
    relative_humidity_from_temperature_dewpoint,
    wind_speed_from_uv,
)
from librewxr.data.weather_sampling import web_mercator_tile_latlons


@dataclass(frozen=True)
class _TileSamplingContext:
    z: int
    x: int
    y: int
    tile_size: int
    padding: int


@runtime_checkable
class NWPSource(Protocol):
    """A numerical weather prediction data source."""

    name: str

    def available_fields(self) -> frozenset[WeatherField]:
        """Return the generic weather fields currently exposed by this source."""
        ...

    def has_field(self, field: WeatherField) -> bool:
        """Whether this source exposes ``field`` through ``sample_field``."""
        ...

    def sample_field(
        self,
        field: WeatherField,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = True,
    ) -> np.ndarray:
        """Return ``field`` in its canonical physical unit at each point.

        Missing values are represented by ``NaN``. Output shape equals
        ``lat.shape``; callers may request bilinear or nearest-neighbour spatial
        sampling, subject to the field specification.
        """
        ...

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = False,
    ) -> np.ndarray:
        """Return uint8 dBZ-encoded precipitation at each (lat, lon) point.

        Encoding matches the radar pipeline: pixel = (dBZ + 32) * 2.
        Output shape == lat.shape.
        """
        ...

    @property
    def supports_snow(self) -> bool:
        """Whether this source can classify precipitation as rain vs. snow.

        Sources that lack a snow-ratio field (e.g. HRRR, DMI DINI, ICON-EU)
        return ``False`` so the chain dispatcher skips their expensive
        ``domain_mask`` and falls through to a source that can (IFS).
        """
        ...

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        """Return bool mask: True where precipitation is snow. Shape == lat.shape."""
        ...

    def domain_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return bool mask: True where this source has coverage. Shape == lat.shape."""
        ...

    def feather_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return float32 mask in [0, 1] for soft chain blending.

        Values close to 1.0 mean "trust this source fully here"; values
        close to 0.0 hand control to the next source in the chain.
        Sources with a hard boundary (e.g., the global IFS fallback)
        return ``domain_mask(lat, lon).astype(float32)``.  Sources with
        a finite domain (HRRR, future regional NWP) return a smooth
        taper to 0 at the boundary so chain blending produces a
        continuous transition instead of a visible seam.
        """
        ...

    def has_data_at(self, timestamp: int) -> bool:
        """Whether this source can answer for the given valid time right now."""
        ...

    def has_data(self) -> bool:
        """Whether this source has any data loaded at all."""
        ...


class NWPChain:
    """Dispatches sample / snow_mask queries across NWP sources in priority order.

    ``sample`` does a soft, weight-accumulating blend across sources.
    Each source contributes ``remaining_weight × its_feather`` of its
    sampled values, with ``remaining`` decreasing as preceding sources
    fill up.  When a source's feather is binary (1 inside / 0 outside,
    e.g. the global IFS fallback) the blend collapses to a hard fill —
    so a chain of binary-feather sources behaves identically to a
    first-fill dispatcher.  When a source's feather tapers smoothly
    near its boundary (e.g. HRRR's LCC edge), the chain produces a
    continuous transition into the next source instead of a visible
    seam.

    ``get_snow_mask`` stays a hard first-fill: blending booleans is
    meaningless and the snow flag is per-pixel categorical.
    """

    def __init__(self, sources: list[NWPSource]):
        self._sources = list(sources)

    @property
    def sources(self) -> list[NWPSource]:
        return list(self._sources)

    def has_data(self) -> bool:
        """True if any registered source has data loaded."""
        return any(src.has_data() for src in self._sources)

    @staticmethod
    def _source_fields(src: NWPSource) -> frozenset[WeatherField]:
        """Return a source's fields, tolerating pre-field third-party sources.

        The compatibility fallback is deliberately precipitation-only. All
        built-in sources implement the complete protocol through the shared
        mixin, but this keeps existing external integrations usable during the
        interface transition.
        """

        available_fields = getattr(src, "available_fields", None)
        if available_fields is None:
            return frozenset({WeatherField.PRECIPITATION})
        return frozenset(WeatherField(field) for field in available_fields())

    @classmethod
    def _source_has_field(cls, src: NWPSource, field: WeatherField) -> bool:
        has_field = getattr(src, "has_field", None)
        if has_field is not None:
            return bool(has_field(field))
        return field in cls._source_fields(src)

    @staticmethod
    def _source_has_data(
        src: NWPSource,
        field: WeatherField,
        timestamp: int | None,
    ) -> bool:
        has_field_at = getattr(src, "has_field_at", None)
        if has_field_at is not None:
            return bool(has_field_at(field, timestamp))
        if timestamp is None:
            return src.has_data()
        return src.has_data_at(timestamp)

    def available_fields(self) -> frozenset[WeatherField]:
        """Return native plus derivable fields exposed by the chain."""

        fields: set[WeatherField] = set()
        for src in self._sources:
            fields.update(self._source_fields(src))

        # A derived field is available only when all its native dependencies
        # can be sampled somewhere in the chain.
        for candidate in WeatherField:
            spec = field_spec(candidate)
            if spec.derived and set(spec.dependencies) <= fields:
                fields.add(candidate)
        return frozenset(fields)

    def has_field(self, field: WeatherField) -> bool:
        """Whether ``field`` can be sampled or derived by this chain."""

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
        """Sample and feather a generic field in canonical physical units.

        Continuous values are blended after source decoding, so encoded nodata
        sentinels and affine storage scales cannot contaminate the result.
        Categorical specifications use hard priority selection instead.
        """

        if lat.shape != lon.shape:
            raise ValueError("lat and lon must have identical shapes")
        return self._sample_field(
            WeatherField(field),
            lat,
            lon,
            timestamp,
            bilinear,
            tile_context=None,
        )

    def sample_tile_field(
        self,
        field: WeatherField,
        z: int,
        x: int,
        y: int,
        timestamp: int | None = None,
        tile_size: int = 256,
        padding: int = 0,
        bilinear: bool = True,
    ) -> np.ndarray:
        """Sample a complete XYZ tile through the same field chain.

        Regular-grid sources may consume the tile context directly and reuse a
        cached spatial plan. Other sources keep using coordinate-array sampling.
        """

        lat, lon = web_mercator_tile_latlons(z, x, y, tile_size, padding)
        context = _TileSamplingContext(z, x, y, tile_size, padding)
        return self._sample_field(
            WeatherField(field),
            lat,
            lon,
            timestamp,
            bilinear,
            tile_context=context,
        )

    def _sample_field(
        self,
        normalized: WeatherField,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None,
        bilinear: bool,
        tile_context: _TileSamplingContext | None,
    ) -> np.ndarray:
        """Shared coordinate/tile implementation for continuous fields."""

        spec = field_spec(normalized)

        if spec.derived:
            if not set(spec.dependencies) <= self.available_fields():
                return np.full(lat.shape, np.nan, dtype=np.float32)
            dependencies = [
                self._sample_field(
                    dependency,
                    lat,
                    lon,
                    timestamp,
                    bilinear,
                    tile_context,
                )
                for dependency in spec.dependencies
            ]
            if normalized is WeatherField.RELATIVE_HUMIDITY_2M:
                return relative_humidity_from_temperature_dewpoint(*dependencies)
            if normalized is WeatherField.WIND_SPEED_10M:
                return wind_speed_from_uv(*dependencies)
            raise ValueError(f"No derivation function for {normalized.value}")

        if spec.categorical:
            return self._sample_categorical_field(
                normalized,
                lat,
                lon,
                timestamp=timestamp,
                tile_context=tile_context,
            )

        weighted_sum = np.zeros(lat.shape, dtype=np.float32)
        weight_sum = np.zeros(lat.shape, dtype=np.float32)
        remaining = np.ones(lat.shape, dtype=np.float32)

        for src in self._sources:
            if not (remaining > 0.0).any():
                break
            if not self._source_has_field(src, normalized):
                continue
            if not self._source_has_data(src, normalized, timestamp):
                continue

            feather = np.asarray(src.feather_mask(lat, lon), dtype=np.float32)
            feather = np.where(np.isfinite(feather), feather, 0.0)
            feather = np.clip(feather, 0.0, 1.0)
            potential_weight = remaining * feather
            relevant = potential_weight > 0.0
            if not relevant.any():
                continue

            sampled = self._sample_source_field(
                src,
                normalized,
                lat,
                lon,
                relevant,
                timestamp,
                bilinear,
                tile_context,
            )
            valid = np.isfinite(sampled)
            if not valid.any():
                continue

            relevant_indices = np.flatnonzero(relevant)
            valid_indices = relevant_indices[valid.reshape(-1)]
            flat_weight = potential_weight.reshape(-1)[valid_indices]
            weighted_sum.reshape(-1)[valid_indices] += (
                flat_weight * sampled.reshape(-1)[valid.reshape(-1)]
            )
            weight_sum.reshape(-1)[valid_indices] += flat_weight
            remaining.reshape(-1)[valid_indices] *= (
                1.0 - feather.reshape(-1)[valid_indices]
            )

        result = np.full(lat.shape, np.nan, dtype=np.float32)
        valid_result = weight_sum > 0.0
        result[valid_result] = weighted_sum[valid_result] / weight_sum[valid_result]
        return result

    @staticmethod
    def _sample_source_field(
        src: NWPSource,
        field: WeatherField,
        lat: np.ndarray,
        lon: np.ndarray,
        relevant: np.ndarray,
        timestamp: int | None,
        bilinear: bool,
        tile_context: _TileSamplingContext | None,
    ) -> np.ndarray:
        """Sample relevant pixels, using a source tile plan when available."""

        tile_sampler = getattr(src, "sample_tile_field", None)
        if tile_context is not None and tile_sampler is not None:
            full = np.asarray(
                tile_sampler(
                    field,
                    tile_context.z,
                    tile_context.x,
                    tile_context.y,
                    timestamp,
                    tile_context.tile_size,
                    tile_context.padding,
                    bilinear,
                ),
                dtype=np.float32,
            )
            if full.shape != lat.shape:
                raise ValueError(
                    f"{src.name} returned tile shape {full.shape}, expected {lat.shape}"
                )
            return full[relevant]

        sampler = getattr(src, "sample_field", None)
        if sampler is None:
            # A transition-only adapter for older precipitation sources.
            encoded = src.sample(
                lat[relevant], lon[relevant], timestamp, bilinear
            )
            return np.asarray(decode_field(field, encoded), dtype=np.float32)
        return np.asarray(
            sampler(
                field,
                lat[relevant],
                lon[relevant],
                timestamp,
                bilinear,
            ),
            dtype=np.float32,
        )

    def _sample_categorical_field(
        self,
        field: WeatherField,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None,
        tile_context: _TileSamplingContext | None = None,
    ) -> np.ndarray:
        """Select categorical data from the first valid source per pixel."""

        result = np.full(lat.shape, np.nan, dtype=np.float32)
        unfilled = np.ones(lat.shape, dtype=bool)
        for src in self._sources:
            if not unfilled.any():
                break
            if not self._source_has_field(src, field):
                continue
            if not self._source_has_data(src, field, timestamp):
                continue
            feather = np.asarray(src.feather_mask(lat, lon), dtype=np.float32)
            covered = unfilled & np.isfinite(feather) & (feather > 0.0)
            if not covered.any():
                continue
            sampled = self._sample_source_field(
                src,
                field,
                lat,
                lon,
                covered,
                timestamp,
                False,
                tile_context,
            )
            valid = np.isfinite(sampled)
            covered_indices = np.flatnonzero(covered)
            valid_indices = covered_indices[valid.reshape(-1)]
            result.reshape(-1)[valid_indices] = sampled.reshape(-1)[valid.reshape(-1)]
            unfilled.reshape(-1)[valid_indices] = False
        return result

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = False,
    ) -> np.ndarray:
        out = np.zeros(lat.shape, dtype=np.float32)
        remaining = np.ones(lat.shape, dtype=np.float32)
        for src in self._sources:
            if not (remaining > 0.0).any():
                break
            if timestamp is not None and not src.has_data_at(timestamp):
                continue
            if timestamp is None and not src.has_data():
                continue
            feather = src.feather_mask(lat, lon).astype(np.float32, copy=False)
            weight = remaining * feather
            relevant = weight > 0.0
            if not relevant.any():
                continue
            sub_lat = lat[relevant]
            sub_lon = lon[relevant]
            sample_vals = src.sample(sub_lat, sub_lon, timestamp, bilinear)
            contribution = np.zeros(lat.shape, dtype=np.float32)
            contribution[relevant] = sample_vals.astype(np.float32, copy=False)
            out += weight * contribution
            remaining *= 1.0 - feather
        # NaN values from out-of-domain LCC projections (HRRR, DINI,
        # WRF-SMN) can flow through the feather-weighted blend into
        # ``out``.  clip + astype on NaN produces a RuntimeWarning;
        # the resulting 0 values are filtered by domain_mask downstream.
        with np.errstate(invalid="ignore"):
            return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        out = np.zeros(lat.shape, dtype=bool)
        unfilled = np.ones(lat.shape, dtype=bool)
        for src in self._sources:
            if not unfilled.any():
                break
            if not src.supports_snow:
                continue
            if timestamp is not None and not src.has_data_at(timestamp):
                continue
            if timestamp is None and not src.has_data():
                continue
            domain = src.domain_mask(lat, lon)
            mask = unfilled & domain
            if not mask.any():
                continue
            sub_lat = lat[mask]
            sub_lon = lon[mask]
            out[mask] = src.get_snow_mask(sub_lat, sub_lon, timestamp)
            unfilled &= ~domain
        return out
