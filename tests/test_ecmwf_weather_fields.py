# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Offline integration tests for global ECMWF physical weather fields."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from librewxr.api import routes
from librewxr.data.master_state import apply_state, dump_state, load_state
from librewxr.data.weather_fields import WeatherField, field_spec
from librewxr.sources.world.ifs import grid as ifs_module
from librewxr.sources.world.ifs.grid import (
    IFS_FIELD_VARIABLES,
    REQUIRED_WEATHER_FIELDS,
    ECMWFGrid,
)
from librewxr.sources.world.ifs.models import WeatherFrame

pytestmark = pytest.mark.ecmwf


class _FakeChild:
    def __init__(self, values: np.ndarray):
        self.values = values
        self.closed = False

    def __getitem__(self, _key):
        return self.values[np.newaxis, :]

    def close(self):
        self.closed = True


class _FakeReader:
    def __init__(self, fields: dict[str, np.ndarray]):
        self.fields = fields
        self.closed = False

    def get_child_by_name(self, name: str):
        if name not in self.fields:
            raise KeyError(name)
        return _FakeChild(self.fields[name])

    def close(self):
        self.closed = True


class _FakeFilesystem:
    def __init__(self, latest: dict, objects: dict[str, dict[str, np.ndarray]]):
        self.latest = latest
        self.objects = objects
        self.metadata_reads = 0
        self.object_opens = 0
        self.object_downloads = 0

    def cat(self, _path: str) -> bytes:
        self.metadata_reads += 1
        return json.dumps(self.latest).encode()

    def get_file(self, remote_path: str, local_path: str) -> None:
        self.object_opens += 1
        self.object_downloads += 1
        Path(local_path).write_text(remote_path.rsplit("/", 1)[-1])


class _FakeOmFileReader:
    objects: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def from_fsspec(fs: _FakeFilesystem, path: str) -> _FakeReader:
        fs.object_opens += 1
        return _FakeReader(fs.objects[path.rsplit("/", 1)[-1]])

    @classmethod
    def from_path(cls, path: str) -> _FakeReader:
        assert isinstance(path, str)
        return _FakeReader(cls.objects[Path(path).read_text()])


def _vt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _object_name(vt: str) -> str:
    return vt.replace("Z", "").replace(":", "") + ".om"


def _run_fixture(base: datetime) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    valid_times = [_vt(base + timedelta(hours=hour)) for hour in range(3)]
    variables = [
        *IFS_FIELD_VARIABLES.values(),
        "precipitation",
        "snowfall_water_equivalent",
    ]
    latest = {
        "completed": True,
        "reference_time": valid_times[0],
        "last_modified_time": _vt(base + timedelta(minutes=45)),
        "valid_times": valid_times,
        "variables": variables,
    }
    objects: dict[str, dict[str, np.ndarray]] = {}
    for index, valid_time in enumerate(valid_times):
        fields = {
            "temperature_2m": np.full(8, 10.0 + index * 10.0, dtype=np.float32),
            "dew_point_2m": np.full(8, 5.0 + index * 5.0, dtype=np.float32),
            "pressure_msl": np.full(8, 101000.0 + index * 1000.0, dtype=np.float32),
            "wind_u_component_10m": np.full(8, 3.0 + index, dtype=np.float32),
            "wind_v_component_10m": np.full(8, 4.0 + index, dtype=np.float32),
        }
        # Open-Meteo omits backward accumulations at T+0.
        if index > 0:
            fields["precipitation"] = np.full(8, float(index), dtype=np.float32)
            fields["snowfall_water_equivalent"] = np.zeros(8, dtype=np.float32)
        objects[_object_name(valid_time)] = fields
    return latest, objects


@pytest.fixture
def loaded_grid(tmp_path: Path, monkeypatch):
    base = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    latest, objects = _run_fixture(base)
    fs = _FakeFilesystem(latest, objects)

    monkeypatch.setattr(ifs_module, "GRID_HEIGHT", 2)
    monkeypatch.setattr(ifs_module, "GRID_WIDTH", 4)
    monkeypatch.setattr(ifs_module, "GRID_SHAPE", (2, 4))
    monkeypatch.setattr(ifs_module, "OmFileReader", _FakeOmFileReader)
    _FakeOmFileReader.objects = objects
    regrid_calls: list[np.ndarray] = []

    def fake_regrid(raw, **_kwargs):
        regrid_calls.append(np.asarray(raw))
        return np.asarray(raw, dtype=np.float32).reshape(2, 4)

    monkeypatch.setattr(ifs_module, "interpolate", fake_regrid)
    monkeypatch.setattr(ifs_module.settings, "ecmwf_interpolation", False)
    monkeypatch.setattr(ifs_module.settings, "weather_fields_forecast_hours", 48)
    monkeypatch.setattr(ifs_module.settings, "weather_fields_max_timesteps", 0)
    monkeypatch.setattr(ifs_module.settings, "nwp_fetch_concurrency", 2)

    grid = ECMWFGrid(cache_dir=tmp_path)
    grid._fs = fs
    assert grid._fetch_sync()
    return grid, fs, base, regrid_calls


def test_loads_all_required_fields_in_compact_separate_memmaps(loaded_grid):
    grid, fs, base, regrid_calls = loaded_grid
    native_times = grid._field_timestamps(WeatherField.TEMPERATURE_2M)

    assert len(native_times) == 3
    assert grid.timestep_count == 2
    assert fs.object_opens == 3
    assert fs.object_downloads == 3
    assert len(regrid_calls) == 3 * 5 + 2 * 2
    for timestamp in native_times:
        frame = grid._timesteps[timestamp]
        assert REQUIRED_WEATHER_FIELDS <= frame.fields.keys()
        for field in REQUIRED_WEATHER_FIELDS:
            values = frame.field(field)
            assert isinstance(values, np.memmap)
            assert values.mode == "r"
            assert values.dtype == field_spec(field).storage_dtype
            filename = Path(values.filename).name
            assert filename.startswith("v2_r")
            assert f"_t{timestamp}_{field.value}.dat" in filename
            assert Path(values.filename).stat().st_size == values.nbytes
    assert not list(grid._memmap_dir.glob("*.tmp"))
    assert not list(grid._memmap_dir.rglob("*.om.tmp"))


def test_units_and_linear_time_interpolation_after_sampling(loaded_grid):
    grid, _fs, base, _calls = loaded_grid
    t0 = int(base.timestamp())
    lat = np.array([90.0])
    lon = np.array([-180.0])

    np.testing.assert_allclose(
        grid.sample_field(WeatherField.TEMPERATURE_2M, lat, lon, t0), [10.0]
    )
    np.testing.assert_allclose(
        grid.sample_field(WeatherField.PRESSURE_MSL, lat, lon, t0), [1010.0]
    )
    np.testing.assert_allclose(
        grid.sample_field(WeatherField.WIND_U_10M, lat, lon, t0), [3.0]
    )
    midpoint = t0 + 1800
    np.testing.assert_allclose(
        grid.sample_field(WeatherField.TEMPERATURE_2M, lat, lon, midpoint),
        [15.0],
    )
    np.testing.assert_allclose(
        grid.sample_field(WeatherField.PRESSURE_MSL, lat, lon, midpoint),
        [1015.0],
    )


def test_time_interpolation_clamps_only_at_available_boundaries(loaded_grid):
    grid, _fs, base, _calls = loaded_grid
    t0 = int(base.timestamp())
    lat = np.array([90.0])
    lon = np.array([-180.0])

    before = grid.sample_field(WeatherField.TEMPERATURE_2M, lat, lon, t0 - 9999)
    after = grid.sample_field(WeatherField.TEMPERATURE_2M, lat, lon, t0 + 99999)
    np.testing.assert_allclose(before, [10.0])
    np.testing.assert_allclose(after, [30.0])


def test_regridding_rolls_zero_to_360_longitudes(monkeypatch):
    monkeypatch.setattr(ifs_module, "GRID_HEIGHT", 2)
    monkeypatch.setattr(ifs_module, "GRID_WIDTH", 4)
    monkeypatch.setattr(ifs_module, "GRID_SHAPE", (2, 4))
    monkeypatch.setattr(
        ifs_module,
        "interpolate",
        lambda _raw, **_kwargs: np.arange(8, dtype=np.float32).reshape(2, 4),
    )
    reader = _FakeReader({"temperature_2m": np.arange(8, dtype=np.float32)})

    result = ECMWFGrid._read_regridded(reader, "temperature_2m")
    np.testing.assert_array_equal(result, [[2, 3, 0, 1], [6, 7, 4, 5]])


def test_weather_window_preserves_published_cadence_and_independent_cap():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    valid_times = [
        _vt(base + timedelta(hours=hour)) for hour in (0, 1, 2, 4, 7, 49)
    ]

    selected = ECMWFGrid._select_weather_valid_times(
        valid_times,
        forecast_hours=48,
        max_timesteps=0,
        now_ts=int(base.timestamp()),
    )
    capped = ECMWFGrid._select_weather_valid_times(
        valid_times,
        forecast_hours=48,
        max_timesteps=3,
        now_ts=int(base.timestamp()),
    )

    assert selected == valid_times[:-1]
    assert capped == valid_times[:3]


def test_unchanged_run_does_not_reopen_om_objects(loaded_grid):
    grid, fs, _base, _calls = loaded_grid
    opens = fs.object_opens
    model_version = grid.model_version

    assert grid._fetch_sync()
    assert fs.object_opens == opens
    assert fs.metadata_reads == 2
    assert grid.model_version == model_version


def test_unchanged_run_refetches_only_timestep_with_missing_memmap(loaded_grid):
    grid, fs, base, _calls = loaded_grid
    timestamp = int(base.timestamp())
    missing = Path(
        grid._timesteps[timestamp]
        .field(WeatherField.TEMPERATURE_2M)
        .filename
    )
    missing.unlink()
    opens = fs.object_opens

    assert grid._fetch_sync()
    assert fs.object_opens == opens + 1
    assert missing.exists()


def test_existing_weather_frame_fetches_only_missing_precipitation(loaded_grid):
    """A future weather frame must not be downloaded and regridded twice."""
    grid, fs, base, regrid_calls = loaded_grid
    timestamp = int((base + timedelta(hours=1)).timestamp())
    original = grid._timesteps[timestamp]
    precipitation = original.field(WeatherField.PRECIPITATION)
    snow = original.snow_mask
    assert snow is not None
    Path(precipitation.filename).unlink()
    Path(snow.filename).unlink()
    grid._timesteps[timestamp] = WeatherFrame(
        timestamp,
        {
            field: original.field(field)
            for field in REQUIRED_WEATHER_FIELDS
        },
        None,
    )
    opens = fs.object_opens
    downloads = fs.object_downloads
    calls = len(regrid_calls)

    assert grid._fetch_sync()

    refreshed = grid._timesteps[timestamp]
    assert fs.object_opens == opens + 1
    assert fs.object_downloads == downloads
    assert len(regrid_calls) == calls + 2
    assert REQUIRED_WEATHER_FIELDS <= refreshed.fields.keys()
    assert refreshed.has_field(WeatherField.PRECIPITATION)
    assert refreshed.snow_mask is not None


def test_incomplete_new_run_is_not_published_and_old_files_survive(loaded_grid):
    grid, fs, base, _calls = loaded_grid
    old_frames = grid._timesteps
    old_reference = grid.reference_time
    old_files = set(grid._memmap_dir.glob("*.dat"))

    new_base = base + timedelta(hours=6)
    latest, objects = _run_fixture(new_base)
    del objects[_object_name(latest["valid_times"][1])]["dew_point_2m"]
    fs.latest = latest
    fs.objects = objects

    assert not grid._fetch_sync()
    assert grid._timesteps is old_frames
    assert grid.reference_time == old_reference
    assert old_files <= set(grid._memmap_dir.glob("*.dat"))
    assert "required timestep preparation failed" in grid._last_update_error


def test_missing_metadata_variable_keeps_active_generation(loaded_grid):
    grid, fs, _base, _calls = loaded_grid
    old_frames = grid._timesteps
    fs.latest = dict(fs.latest)
    fs.latest["reference_time"] = "2099-01-01T00:00:00Z"
    fs.latest["variables"] = [
        name for name in fs.latest["variables"] if name != "pressure_msl"
    ]
    opens = fs.object_opens

    assert not grid._fetch_sync()
    assert grid._timesteps is old_frames
    assert fs.object_opens == opens
    assert "pressure_msl" in grid._last_update_error


def test_state_roundtrip_reopens_every_field_read_only(loaded_grid, tmp_path):
    producer, _fs, base, _calls = loaded_grid
    dump_state({"ecmwf_grid": producer}, tmp_path)
    payload = load_state(tmp_path)
    consumer = ECMWFGrid(cache_dir=tmp_path)

    assert payload is not None
    assert apply_state(payload, {"ecmwf_grid": consumer}) == ["ecmwf_grid"]
    assert consumer.reference_time == producer.reference_time
    frame = consumer._timesteps[int(base.timestamp())]
    assert REQUIRED_WEATHER_FIELDS <= frame.fields.keys()
    assert all(isinstance(values, np.memmap) for values in frame.fields.values())
    assert all(values.mode == "r" for values in frame.fields.values())


def test_stale_master_snapshot_cannot_replace_newer_active_manifest(
    loaded_grid, tmp_path
):
    producer, _fs, base, _calls = loaded_grid
    stale = {
        "memmap_dir": str(producer._memmap_dir),
        "reference_time": producer.reference_time,
        "timesteps": {},
    }
    consumer = ECMWFGrid(cache_dir=tmp_path)

    assert consumer._content_version == producer._content_version == 1
    assert consumer.available_timestamps()
    consumer.__setstate__(stale)

    assert consumer._content_version == 1
    assert consumer.available_timestamps() == producer.available_timestamps()
    assert int(base.timestamp()) in consumer._timesteps


def test_state_descriptor_preserves_run_path_through_cache_symlink(tmp_path):
    real_cache = tmp_path / "real-cache"
    real_cache.mkdir()
    linked_cache = tmp_path / "linked-cache"
    linked_cache.symlink_to(real_cache, target_is_directory=True)
    grid = ECMWFGrid(cache_dir=linked_cache)
    run_file = (
        real_cache
        / "ecmwf_ifs"
        / "runs"
        / "20260806T060000Z"
        / "temperature_2m.dat"
    )

    assert grid._relative_memmap_name(run_file) == (
        "runs/20260806T060000Z/temperature_2m.dat"
    )


def test_legacy_snapshot_is_backward_compatible(tmp_path):
    producer = ECMWFGrid(cache_dir=tmp_path)
    timestamp = 1700000000
    precip = producer._to_memmap(
        "legacy_precip", np.full((2, 4), 84, dtype=np.uint8)
    )
    snow = producer._to_memmap(
        "legacy_snow", np.zeros((2, 4), dtype=bool)
    )
    legacy = {
        "memmap_dir": str(producer._memmap_dir),
        "reference_time": "2023-11-14T00:00:00Z",
        "timesteps": {
            str(timestamp): {
                "precip": [Path(precip.filename).name, precip.dtype.str, [2, 4]],
                "snow": [Path(snow.filename).name, snow.dtype.str, [2, 4]],
            }
        },
    }
    consumer = ECMWFGrid(cache_dir=tmp_path)

    old_state_json = {
        "version": 1,
        "written_at": timestamp,
        "stores": {"ecmwf_grid": json.loads(json.dumps(legacy))},
    }
    assert apply_state(old_state_json, {"ecmwf_grid": consumer}) == ["ecmwf_grid"]
    assert consumer.timestep_count == 1
    assert consumer._timesteps[timestamp].has_field(WeatherField.PRECIPITATION)
    assert consumer._last_update_error is None


def test_cleanup_removes_only_unretained_memmaps(loaded_grid):
    grid, _fs, _base, _calls = loaded_grid
    active_files = grid._frame_files(grid._timesteps.values())
    stale = grid._memmap_dir / "v1_old_run_stale.dat"
    stale.write_bytes(b"stale")

    grid._cleanup_memmap_files(keep=active_files)
    assert not stale.exists()
    assert all((grid._memmap_dir / name).exists() for name in active_files)


def test_health_block_reports_global_field_generation(loaded_grid, monkeypatch):
    grid, _fs, _base, _calls = loaded_grid
    monkeypatch.setattr(routes, "nwp_grids", {"ecmwf_grid": grid})

    block = routes._nwp_grid_health_blocks()["ecmwf_grid"]
    assert block["active_model_run"] == grid.reference_time
    assert block["valid_times"] == 3
    assert block["oldest_valid_time"] < block["latest_valid_time"]
    assert set(field.value for field in REQUIRED_WEATHER_FIELDS) <= set(
        block["loaded_fields"]
    )
    assert block["field_bytes"]["temperature_2m"] > 0
    assert block["run_age_seconds"] is not None
    assert block["stale"] is False
    assert block["last_update_error"] is None
