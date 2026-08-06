# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for generic weather-field metadata, codecs, and derivations."""

import numpy as np
import pytest

from librewxr.data.weather_fields import (
    FIELD_SPECS,
    WeatherField,
    WeatherFieldSourceMixin,
    decode_field,
    encode_field,
    field_spec,
    relative_humidity_from_temperature_dewpoint,
    wind_direction_from_uv,
    wind_speed_from_uv,
)


@pytest.mark.parametrize(
    ("field", "values", "expected_dtype", "tolerance"),
    [
        (WeatherField.PRECIPITATION, [-32.0, 12.5, 95.5], np.uint8, 0.25),
        (WeatherField.TEMPERATURE_2M, [-40.0, 18.7, 60.0], np.int16, 0.05),
        (WeatherField.DEWPOINT_2M, [-70.0, 7.3, 50.0], np.int16, 0.05),
        (WeatherField.RELATIVE_HUMIDITY_2M, [0.0, 54.0, 100.0], np.uint8, 0.5),
        (WeatherField.PRESSURE_MSL, [800.0, 1013.2, 1100.0], np.uint16, 0.05),
        (WeatherField.WIND_U_10M, [-80.0, 3.4, 150.0], np.int16, 0.05),
        (WeatherField.WIND_V_10M, [-150.0, -2.7, 80.0], np.int16, 0.05),
        (WeatherField.WIND_SPEED_10M, [0.0, 12.3, 150.0], np.uint16, 0.05),
    ],
)
def test_encode_decode_round_trip(field, values, expected_dtype, tolerance):
    physical = np.asarray(values, dtype=np.float32)
    encoded = encode_field(field, physical)

    assert encoded.dtype == np.dtype(expected_dtype)
    np.testing.assert_allclose(decode_field(field, encoded), physical, atol=tolerance)


@pytest.mark.parametrize(
    ("field", "values", "expected"),
    [
        (WeatherField.TEMPERATURE_2M, [-101.0, 61.0], [-100.0, 60.0]),
        (WeatherField.PRESSURE_MSL, [700.0, 1200.0], [800.0, 1100.0]),
        (WeatherField.WIND_U_10M, [-200.0, 200.0], [-150.0, 150.0]),
        (WeatherField.RELATIVE_HUMIDITY_2M, [-1.0, 101.0], [0.0, 100.0]),
    ],
)
def test_encode_clips_to_physical_range(field, values, expected):
    decoded = decode_field(field, encode_field(field, np.asarray(values)))
    np.testing.assert_allclose(decoded, expected)


def test_encode_can_reject_out_of_range_values():
    with pytest.raises(ValueError, match="temperature_2m values must be within"):
        encode_field(WeatherField.TEMPERATURE_2M, np.array([61.0]), clip=False)


@pytest.mark.parametrize(
    "field",
    [
        WeatherField.TEMPERATURE_2M,
        WeatherField.DEWPOINT_2M,
        WeatherField.RELATIVE_HUMIDITY_2M,
        WeatherField.PRESSURE_MSL,
        WeatherField.WIND_U_10M,
        WeatherField.WIND_V_10M,
        WeatherField.WIND_SPEED_10M,
    ],
)
def test_nodata_round_trip(field):
    encoded = encode_field(field, np.array([1.0, np.nan]))

    assert encoded[1] == field_spec(field).nodata
    decoded = decode_field(field, encoded)
    assert np.isfinite(decoded[0])
    assert np.isnan(decoded[1])


def test_precipitation_preserves_legacy_encoding_and_rejects_nodata():
    encoded = encode_field(WeatherField.PRECIPITATION, np.array([-32.0, 0.0, 12.5]))
    np.testing.assert_array_equal(encoded, np.array([0, 64, 89], dtype=np.uint8))

    with pytest.raises(ValueError, match="has no nodata representation"):
        encode_field(WeatherField.PRECIPITATION, np.array([np.nan]))


def test_registry_contains_one_immutable_spec_per_field():
    assert frozenset(FIELD_SPECS) == frozenset(WeatherField)
    assert all(spec.field is field for field, spec in FIELD_SPECS.items())
    with pytest.raises(TypeError):
        FIELD_SPECS[WeatherField.TEMPERATURE_2M] = field_spec(  # type: ignore[index]
            WeatherField.TEMPERATURE_2M
        )


def test_relative_humidity_uses_temperature_and_dewpoint():
    humidity = relative_humidity_from_temperature_dewpoint(
        np.array([20.0, 20.0, 10.0, np.nan]),
        np.array([20.0, 10.0, 15.0, 0.0]),
    )

    assert humidity.dtype == np.float32
    np.testing.assert_allclose(humidity[:2], [100.0, 52.54], atol=0.1)
    assert humidity[2] == 100.0
    assert np.isnan(humidity[3])


def test_wind_speed_and_optional_direction_from_components():
    speed = wind_speed_from_uv(np.array([3.0, 0.0, np.nan]), [4.0, 0.0, 1.0])
    direction = wind_direction_from_uv(np.array([0.0, -1.0, 0.0]), [-1.0, 0.0, 0.0])

    np.testing.assert_allclose(speed[:2], [5.0, 0.0])
    assert np.isnan(speed[2])
    np.testing.assert_allclose(direction[:2], [0.0, 90.0])
    assert np.isnan(direction[2])


class _LegacyPrecipitationGrid(WeatherFieldSourceMixin):
    name = "legacy"

    def sample(self, lat, lon, timestamp=None, bilinear=False):
        return np.full(lat.shape, 64, dtype=np.uint8)


def test_existing_source_mixin_declares_only_precipitation():
    source = _LegacyPrecipitationGrid()

    assert source.available_fields() == frozenset({WeatherField.PRECIPITATION})
    assert source.has_field(WeatherField.PRECIPITATION)
    assert not source.has_field(WeatherField.TEMPERATURE_2M)
    np.testing.assert_array_equal(
        source.sample_field(
            WeatherField.PRECIPITATION,
            np.array([0.0]),
            np.array([0.0]),
        ),
        np.array([0.0], dtype=np.float32),
    )
    with pytest.raises(KeyError, match="does not provide temperature_2m"):
        source.sample_field(
            WeatherField.TEMPERATURE_2M,
            np.array([0.0]),
            np.array([0.0]),
        )
