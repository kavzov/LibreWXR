# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the NWPChain dispatcher.

The Tier 2 precip-bbox fast-path surface (``_probe_domain_bbox`` /
``_bbox_intersects`` / ``has_precip_in_bbox`` / ``_domain_bboxes``) was
removed when the stitched global precip mask (``PrecipMaskStore``)
replaced it — see ``tests/test_precip_mask.py``.  What remains here is
the dispatcher behaviour that is independent of that gate.
"""

import numpy as np

from librewxr.data.nwp_source import NWPChain, NWPSource
from librewxr.data.weather_fields import WeatherField, WeatherFieldSourceMixin


class FakeRegionalSource:
    """A regional NWP source: finite projection-only domain."""

    def __init__(self, west, south, east, north):
        self._bbox = (west, south, east, north)

    @property
    def name(self) -> str:
        return "fake_regional"

    def domain_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        w, s, e, n = self._bbox
        return (
            (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
        )

    def feather_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return self.domain_mask(lat, lon).astype(np.float32)

    def has_data(self) -> bool:
        return True

    def has_data_at(self, timestamp: int) -> bool:
        return True

    def sample(self, lat, lon, timestamp=None, bilinear=False) -> np.ndarray:
        return np.zeros(lat.shape, dtype=np.uint8)

    @property
    def supports_snow(self) -> bool:
        return False

    def get_snow_mask(self, lat, lon, timestamp=None) -> np.ndarray:
        return np.zeros(lat.shape, dtype=bool)


class FakeFieldSource(WeatherFieldSourceMixin):
    """Configurable physical-field source used to exercise chain blending."""

    def __init__(self, name, fields, values, feather=1.0, legacy_value=0):
        self.name = name
        self._fields = frozenset(fields)
        self._values = values
        self._feather = feather
        self._legacy_value = legacy_value

    def available_fields(self) -> frozenset[WeatherField]:
        return self._fields

    def sample_field(self, field, lat, lon, timestamp=None, bilinear=True):
        if not self.has_field(field):
            raise KeyError(field)
        value = self._values[WeatherField(field)]
        if callable(value):
            return np.asarray(value(lat, lon), dtype=np.float32)
        return np.full(lat.shape, value, dtype=np.float32)

    def sample(self, lat, lon, timestamp=None, bilinear=False):
        return np.full(lat.shape, self._legacy_value, dtype=np.uint8)

    def domain_mask(self, lat, lon):
        return self.feather_mask(lat, lon) > 0.0

    def feather_mask(self, lat, lon):
        if callable(self._feather):
            return np.asarray(self._feather(lat, lon), dtype=np.float32)
        return np.full(lat.shape, self._feather, dtype=np.float32)

    def has_data(self):
        return True

    def has_data_at(self, timestamp):
        return True

    @property
    def supports_snow(self):
        return False

    def get_snow_mask(self, lat, lon, timestamp=None):
        return np.zeros(lat.shape, dtype=bool)


class TestNWPChain:
    def test_sources_property_unchanged(self):
        chain = NWPChain([FakeRegionalSource(-125.0, 25.0, -70.0, 50.0)])
        assert len(chain.sources) == 1
        assert chain.sources[0].name == "fake_regional"

    def test_builtin_field_source_satisfies_extended_protocol(self):
        source = FakeFieldSource(
            "generic",
            {WeatherField.PRECIPITATION},
            {WeatherField.PRECIPITATION: 0.0},
        )
        assert isinstance(source, NWPSource)

    def test_available_fields_include_derivable_fields(self):
        native_fields = {
            WeatherField.TEMPERATURE_2M,
            WeatherField.DEWPOINT_2M,
            WeatherField.WIND_U_10M,
            WeatherField.WIND_V_10M,
        }
        source = FakeFieldSource(
            "global",
            native_fields,
            {field: 1.0 for field in native_fields},
        )
        chain = NWPChain([source])

        assert native_fields <= chain.available_fields()
        assert chain.has_field(WeatherField.RELATIVE_HUMIDITY_2M)
        assert chain.has_field(WeatherField.WIND_SPEED_10M)
        assert not chain.has_field(WeatherField.PRESSURE_MSL)

    def test_sample_field_skips_sources_without_requested_field(self):
        precipitation_only = FakeFieldSource(
            "regional",
            {WeatherField.PRECIPITATION},
            {WeatherField.PRECIPITATION: 50.0},
        )
        global_temperature = FakeFieldSource(
            "global",
            {WeatherField.TEMPERATURE_2M},
            {WeatherField.TEMPERATURE_2M: 22.0},
        )
        chain = NWPChain([precipitation_only, global_temperature])

        sampled = chain.sample_field(
            WeatherField.TEMPERATURE_2M,
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )
        np.testing.assert_allclose(sampled, [22.0, 22.0])

    def test_sample_field_feather_blends_in_physical_units(self):
        lat = np.array([0.0, 0.0])
        lon = np.array([0.0, 1.0])
        regional = FakeFieldSource(
            "regional",
            {WeatherField.TEMPERATURE_2M},
            {WeatherField.TEMPERATURE_2M: 10.0},
            feather=lambda lat, lon: np.array([0.75, 0.25]),
        )
        global_fallback = FakeFieldSource(
            "global",
            {WeatherField.TEMPERATURE_2M},
            {WeatherField.TEMPERATURE_2M: 30.0},
        )

        sampled = NWPChain([regional, global_fallback]).sample_field(
            WeatherField.TEMPERATURE_2M, lat, lon
        )
        np.testing.assert_allclose(sampled, [15.0, 25.0])

    def test_sample_field_nodata_falls_through_to_global_source(self):
        lat = np.array([0.0, 0.0])
        lon = np.array([0.0, 1.0])
        regional = FakeFieldSource(
            "regional",
            {WeatherField.PRESSURE_MSL},
            {
                WeatherField.PRESSURE_MSL: lambda lat, lon: np.array(
                    [1000.0, np.nan]
                )
            },
        )
        global_fallback = FakeFieldSource(
            "global",
            {WeatherField.PRESSURE_MSL},
            {WeatherField.PRESSURE_MSL: 1015.0},
        )

        sampled = NWPChain([regional, global_fallback]).sample_field(
            WeatherField.PRESSURE_MSL, lat, lon
        )
        np.testing.assert_allclose(sampled, [1000.0, 1015.0])

    def test_sample_field_derives_wind_speed_after_component_fallback(self):
        global_wind = FakeFieldSource(
            "global",
            {WeatherField.WIND_U_10M, WeatherField.WIND_V_10M},
            {WeatherField.WIND_U_10M: 3.0, WeatherField.WIND_V_10M: 4.0},
        )
        speed = NWPChain([global_wind]).sample_field(
            WeatherField.WIND_SPEED_10M,
            np.array([0.0]),
            np.array([0.0]),
        )
        np.testing.assert_allclose(speed, [5.0])

    def test_legacy_precipitation_sample_result_is_unchanged(self):
        lat = np.zeros(3)
        lon = np.arange(3, dtype=np.float64)
        regional = FakeFieldSource(
            "regional",
            {WeatherField.PRECIPITATION},
            {WeatherField.PRECIPITATION: 18.0},
            feather=lambda lat, lon: np.array([1.0, 0.25, 0.0]),
            legacy_value=100,
        )
        global_fallback = FakeFieldSource(
            "global",
            {WeatherField.PRECIPITATION},
            {WeatherField.PRECIPITATION: 68.0},
            legacy_value=200,
        )

        sampled = NWPChain([regional, global_fallback]).sample(lat, lon)
        np.testing.assert_array_equal(sampled, np.array([100, 175, 200], dtype=np.uint8))
