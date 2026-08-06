# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for tile-oriented global weather-field sampling."""

from __future__ import annotations

import numpy as np
import pytest

from librewxr.data.nwp_source import NWPChain
from librewxr.data.weather_fields import (
    WeatherField,
    WeatherFieldSourceMixin,
    encode_field,
)
from librewxr.data.weather_sampling import (
    SamplingPlan,
    cached_regular_tile_sampling_plan,
    clear_sampling_plan_cache,
    web_mercator_tile_latlons,
)
from librewxr.sources.world.ifs import grid as ifs_module
from librewxr.sources.world.ifs.grid import ECMWFGrid
from librewxr.sources.world.ifs.models import WeatherFrame

pytestmark = pytest.mark.ecmwf


@pytest.fixture
def regular_global_grid(tmp_path, monkeypatch):
    """Small 90-degree global IFS grid with two complete native frames."""

    monkeypatch.setattr(ifs_module, "PIXEL_SIZE", 90.0)
    monkeypatch.setattr(ifs_module, "GRID_WIDTH", 4)
    monkeypatch.setattr(ifs_module, "GRID_HEIGHT", 3)
    monkeypatch.setattr(ifs_module, "GRID_SHAPE", (3, 4))
    clear_sampling_plan_cache()

    temperature = np.array(
        [[0.0, 10.0, 20.0, 30.0],
         [10.0, 20.0, 30.0, 40.0],
         [20.0, 30.0, 40.0, 50.0]],
        dtype=np.float32,
    )
    fields0 = {
        WeatherField.PRECIPITATION: np.full((3, 4), 96, dtype=np.uint8),
        WeatherField.TEMPERATURE_2M: encode_field(
            WeatherField.TEMPERATURE_2M, temperature
        ),
        WeatherField.DEWPOINT_2M: encode_field(
            WeatherField.DEWPOINT_2M, temperature - 5.0
        ),
        WeatherField.PRESSURE_MSL: encode_field(
            WeatherField.PRESSURE_MSL,
            np.full((3, 4), 1000.0, dtype=np.float32),
        ),
        WeatherField.WIND_U_10M: encode_field(
            WeatherField.WIND_U_10M,
            np.full((3, 4), 3.0, dtype=np.float32),
        ),
        WeatherField.WIND_V_10M: encode_field(
            WeatherField.WIND_V_10M,
            np.full((3, 4), 4.0, dtype=np.float32),
        ),
    }
    fields1 = dict(fields0)
    fields1[WeatherField.TEMPERATURE_2M] = encode_field(
        WeatherField.TEMPERATURE_2M, temperature + 10.0
    )
    fields1[WeatherField.DEWPOINT_2M] = encode_field(
        WeatherField.DEWPOINT_2M, temperature + 5.0
    )

    grid = ECMWFGrid(cache_dir=tmp_path)
    grid._timesteps[0] = WeatherFrame(0, fields0)
    grid._timesteps[3600] = WeatherFrame(3600, fields1)
    yield grid
    clear_sampling_plan_cache()


def test_global_ifs_covers_z0_and_all_continuous_public_fields(
    regular_global_grid,
):
    grid = regular_global_grid
    expected = {
        WeatherField.TEMPERATURE_2M,
        WeatherField.DEWPOINT_2M,
        WeatherField.RELATIVE_HUMIDITY_2M,
        WeatherField.PRESSURE_MSL,
        WeatherField.WIND_U_10M,
        WeatherField.WIND_V_10M,
        WeatherField.WIND_SPEED_10M,
    }

    assert expected <= grid.available_fields()
    for field in expected:
        sampled = grid.sample_tile_field(field, 0, 0, 0, timestamp=999_999, tile_size=16)
        assert sampled.shape == (16, 16)
        assert np.isfinite(sampled).all()
    np.testing.assert_allclose(
        grid.sample_tile_field(
            WeatherField.WIND_SPEED_10M, 0, 0, 0, tile_size=4
        ),
        5.0,
    )


def test_domain_mask_rejects_only_invalid_global_coordinates(regular_global_grid):
    lat = np.array([0.0, 85.0511, -85.0511, np.nan, 91.0])
    lon = np.array([-180.0, 180.0, 540.0, 0.0, 0.0])

    np.testing.assert_array_equal(
        regular_global_grid.domain_mask(lat, lon),
        [True, True, True, False, False],
    )


def test_dateline_wraps_bilinear_sampling(regular_global_grid):
    grid = regular_global_grid
    lat = np.zeros(4)
    lon = np.array([-180.0, 180.0, -180.001, 179.999])

    sampled = grid.sample_field(
        WeatherField.TEMPERATURE_2M, lat, lon, timestamp=0
    )
    np.testing.assert_allclose(sampled[0], sampled[1], atol=1e-6)
    np.testing.assert_allclose(sampled[2], sampled[3], atol=1e-5)


@pytest.mark.parametrize("y", [0, 3])
def test_web_mercator_north_and_south_limits_are_valid(regular_global_grid, y):
    plan = regular_global_grid.sampling_plan(2, 1, y, tile_size=16)

    assert plan.valid.all()
    assert plan.r0.min() >= 0
    assert plan.r1.max() < 3
    assert plan.c0.min() >= 0
    assert plan.c1.max() < 4


def test_tile_padding_and_edges_have_expected_geometry(regular_global_grid):
    plan = regular_global_grid.sampling_plan(0, 0, 0, tile_size=8, padding=2)
    lat, lon = web_mercator_tile_latlons(0, 0, 0, 8, 2)

    assert isinstance(plan, SamplingPlan)
    assert plan.shape == (12, 12)
    assert lat.shape == lon.shape == plan.shape
    assert plan.valid.all()
    assert lon[0, 0] < -180.0
    assert lon[0, -1] > 180.0


def test_bilinear_sampling_uses_four_cells(regular_global_grid):
    sampled = regular_global_grid.sample_field(
        WeatherField.TEMPERATURE_2M,
        np.array([45.0]),
        np.array([-135.0]),
        timestamp=0,
        bilinear=True,
    )

    np.testing.assert_allclose(sampled, [10.0])


def test_temporal_interpolation_samples_before_interpolating(regular_global_grid):
    sampled = regular_global_grid.sample_field(
        WeatherField.TEMPERATURE_2M,
        np.array([45.0]),
        np.array([-135.0]),
        timestamp=1800,
        bilinear=True,
    )

    np.testing.assert_allclose(sampled, [15.0])


def test_sampling_plan_reused_across_fields_and_times(regular_global_grid):
    grid = regular_global_grid
    plan = grid.sampling_plan(1, 0, 0, tile_size=8, padding=1)
    before = cached_regular_tile_sampling_plan.cache_info()

    grid.sample_tile_field(
        WeatherField.TEMPERATURE_2M, 1, 0, 0, 0, 8, 1
    )
    grid.sample_tile_field(
        WeatherField.PRESSURE_MSL, 1, 0, 0, 1800, 8, 1
    )
    after = cached_regular_tile_sampling_plan.cache_info()

    assert grid.sampling_plan(1, 0, 0, 8, 1) is plan
    assert after.hits >= before.hits + 2
    for array in (
        plan.r0, plan.r1, plan.c0, plan.c1, plan.dr, plan.dc, plan.valid
    ):
        assert not isinstance(array, np.memmap)


def test_sampling_plan_invalidated_by_grid_version(regular_global_grid):
    grid = regular_global_grid
    old_version = grid.grid_version
    old_plan = grid.sampling_plan(1, 0, 0, tile_size=8)

    grid.invalidate_sampling_plans()
    new_plan = grid.sampling_plan(1, 0, 0, tile_size=8)

    assert grid.grid_version == old_version + 1
    assert new_plan is not old_plan


def test_adjacent_tiles_have_identical_overlap_samples(regular_global_grid):
    grid = regular_global_grid
    left = grid.sample_tile_field(
        WeatherField.TEMPERATURE_2M, 1, 0, 0, 0, 16, 1
    )
    right = grid.sample_tile_field(
        WeatherField.TEMPERATURE_2M, 1, 1, 0, 0, 16, 1
    )

    np.testing.assert_allclose(left[:, -2], right[:, 0], atol=1e-6)
    np.testing.assert_allclose(left[:, -1], right[:, 1], atol=1e-6)


class _RegionalFieldSource(WeatherFieldSourceMixin):
    name = "regional_fields"

    def __init__(self, fields, value, *, current=True):
        self._fields = frozenset(fields)
        self.value = value
        self.current = current
        self.calls = 0

    def available_fields(self):
        return self._fields

    def sample_field(self, field, lat, lon, timestamp=None, bilinear=True):
        self.calls += 1
        return np.full(lat.shape, self.value, dtype=np.float32)

    def sample(self, lat, lon, timestamp=None, bilinear=False):
        self.calls += 1
        return np.full(lat.shape, 100, dtype=np.uint8)

    def feather_mask(self, lat, lon):
        return np.clip(1.0 - np.abs(lon) / 90.0, 0.0, 1.0).astype(np.float32)

    def domain_mask(self, lat, lon):
        return self.feather_mask(lat, lon) > 0.0

    def has_data(self):
        return self.current

    def has_data_at(self, timestamp):
        return self.current

    @property
    def supports_snow(self):
        return False

    def get_snow_mask(self, lat, lon, timestamp=None):
        return np.zeros(lat.shape, dtype=bool)


def test_regional_field_feathers_to_global_ifs_without_gaps(regular_global_grid):
    grid = regular_global_grid
    regional = _RegionalFieldSource({WeatherField.PRESSURE_MSL}, 900.0)
    precipitation_only = _RegionalFieldSource(
        {WeatherField.PRECIPITATION}, 99.0
    )
    chain = NWPChain([precipitation_only, regional, grid])

    sampled = chain.sample_tile_field(
        WeatherField.PRESSURE_MSL,
        0,
        0,
        0,
        timestamp=1800,
        tile_size=32,
    )

    assert np.isfinite(sampled).all()
    assert sampled.min() >= 900.0
    assert sampled.max() <= 1000.001
    assert sampled.max() > sampled.min()
    assert regional.calls == 1
    assert precipitation_only.calls == 0


def test_outdated_regional_field_falls_through_to_ifs(regular_global_grid):
    regional = _RegionalFieldSource(
        {WeatherField.PRESSURE_MSL}, 900.0, current=False
    )
    chain = NWPChain([regional, regular_global_grid])

    sampled = chain.sample_tile_field(
        WeatherField.PRESSURE_MSL, 0, 0, 0, timestamp=1800, tile_size=8
    )

    np.testing.assert_allclose(sampled, 1000.0)
    assert regional.calls == 0
