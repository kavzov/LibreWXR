# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Production lifecycle tests for persistent global weather generations."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from librewxr.api import routes
from librewxr.data.master_state import apply_state, dump_state, load_state
from librewxr.data.fetcher import RadarFetcher
from librewxr.data.nwp_source import NWPChain
from librewxr.data.weather_fields import WeatherField, encode_field
from librewxr.main import _compute_cache_invalidation, _drop_absent_stores
from librewxr.sources.world.ifs import grid as ifs_module
from librewxr.sources.world.ifs.grid import ECMWFGrid
from librewxr.sources.world.ifs.models import WeatherFrame
from librewxr.sources._base import NWPContribution
from librewxr.tiles.cache import TileCache


def _physical_fields(value: float) -> dict[WeatherField, np.ndarray]:
    shape = (3, 4)
    return {
        WeatherField.PRECIPITATION: np.full(shape, 96, dtype=np.uint8),
        WeatherField.TEMPERATURE_2M: encode_field(
            WeatherField.TEMPERATURE_2M, np.full(shape, value)
        ),
        WeatherField.DEWPOINT_2M: encode_field(
            WeatherField.DEWPOINT_2M, np.full(shape, value - 5.0)
        ),
        WeatherField.PRESSURE_MSL: encode_field(
            WeatherField.PRESSURE_MSL, np.full(shape, 1000.0 + value)
        ),
        WeatherField.WIND_U_10M: encode_field(
            WeatherField.WIND_U_10M, np.full(shape, 3.0)
        ),
        WeatherField.WIND_V_10M: encode_field(
            WeatherField.WIND_V_10M, np.full(shape, 4.0)
        ),
    }


def _prepare_run(
    grid: ECMWFGrid,
    reference_time: str,
    value: float,
) -> dict[int, WeatherFrame]:
    frames = {}
    for timestamp in (1_700_000_000, 1_700_003_600):
        frame = WeatherFrame(timestamp, _physical_fields(value))
        frames[timestamp] = grid._persist_frame(reference_time, frame)
    return frames


def _small_grid(monkeypatch) -> None:
    monkeypatch.setattr(ifs_module, "PIXEL_SIZE", 90.0)
    monkeypatch.setattr(ifs_module, "GRID_WIDTH", 4)
    monkeypatch.setattr(ifs_module, "GRID_HEIGHT", 3)
    monkeypatch.setattr(ifs_module, "GRID_SHAPE", (3, 4))


def test_pipeline_worker_reload_changes_weather_etag_and_preserves_other_cache(
    tmp_path,
    monkeypatch,
):
    _small_grid(monkeypatch)
    run_a = "2026-08-06T00:00:00Z"
    run_b = "2026-08-06T06:00:00Z"
    producer = ECMWFGrid(cache_dir=tmp_path)
    assert producer._publish_prepared(
        _prepare_run(producer, run_a, 10.0),
        reference_time=run_a,
        last_modified_time=run_a,
    )
    dump_state({"ecmwf_grid": producer}, tmp_path)
    first_payload = load_state(tmp_path)
    assert first_payload is not None

    worker = ECMWFGrid(cache_dir=tmp_path, cleanup_tmp=False)
    stores = {"ecmwf_grid": worker}
    assert apply_state(first_payload, stores) == ["ecmwf_grid"]
    cache = TileCache(max_mb=10)
    chain = NWPChain([worker])

    previous_routes = (routes.ecmwf_grid, routes.nwp_chain, routes.tile_cache)
    routes.ecmwf_grid = worker
    routes.nwp_chain = chain
    routes.tile_cache = cache
    app = FastAPI()
    app.include_router(routes.router)
    url = (
        "/v2/weather/temperature_2m/1700001800/256/0/0/0/temperature.png"
    )
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            first = client.get(url)
            assert first.status_code == 200
            first_rgba = np.asarray(
                Image.open(io.BytesIO(first.content)).convert("RGBA")
            )

            cache.put(("sat", "unrelated"), b"satellite")
            cache.put(("cov", "unrelated"), b"coverage")
            cache.put((1_700_000_000, "radar"), b"radar")

            assert producer._publish_prepared(
                _prepare_run(producer, run_b, 20.0),
                reference_time=run_b,
                last_modified_time=run_b,
            )
            dump_state({"ecmwf_grid": producer}, tmp_path)
            second_payload = load_state(tmp_path)
            assert second_payload is not None
            assert apply_state(
                second_payload, stores, prev_payload=first_payload
            ) == ["ecmwf_grid"]

            timestamps, nwp_changed = _compute_cache_invalidation(
                first_payload, second_payload
            )
            assert timestamps is None
            assert nwp_changed is True
            cache.invalidate_nwp_dependent()
            assert cache.get(("sat", "unrelated")) == b"satellite"
            assert cache.get(("cov", "unrelated")) == b"coverage"
            assert cache.get((1_700_000_000, "radar")) is None

            second = client.get(url)
            second_rgba = np.asarray(
                Image.open(io.BytesIO(second.content)).convert("RGBA")
            )
            assert second.status_code == 200
            assert second.headers["etag"] != first.headers["etag"]
            assert not np.array_equal(first_rgba, second_rgba)
    finally:
        routes.ecmwf_grid, routes.nwp_chain, routes.tile_cache = previous_routes

    assert worker.reference_time == run_b
    assert all(
        values.mode == "r"
        for frame in worker._timesteps.values()
        for values in frame.fields.values()
    )
    # The worker may still hold run A while the pipeline publishes B.
    assert (producer._runs_dir / producer._run_key(run_a)).exists()
    assert (producer._runs_dir / producer._run_key(run_b)).exists()


def test_active_manifest_restart_outage_and_run_cleanup(tmp_path, monkeypatch):
    _small_grid(monkeypatch)
    runs = [
        "2026-08-06T00:00:00Z",
        "2026-08-06T06:00:00Z",
        "2026-08-06T12:00:00Z",
    ]
    producer = ECMWFGrid(cache_dir=tmp_path)
    for index, reference_time in enumerate(runs):
        assert producer._publish_prepared(
            _prepare_run(producer, reference_time, 10.0 + index),
            reference_time=reference_time,
            last_modified_time=reference_time,
        )

    manifest = json.loads(producer._manifest_path.read_text(encoding="utf-8"))
    assert manifest["active_run"] == runs[-1]
    assert manifest["previous_run"] == runs[-2]
    assert not (producer._runs_dir / producer._run_key(runs[0])).exists()
    assert (producer._runs_dir / producer._run_key(runs[1])).exists()
    assert (producer._runs_dir / producer._run_key(runs[2])).exists()

    restarted = ECMWFGrid(cache_dir=tmp_path)
    assert restarted.reference_time == runs[-1]
    assert restarted.previous_model_run == runs[-2]
    before = restarted._timesteps
    assert restarted._fail_update("simulated upstream outage") is False
    assert restarted._timesteps is before
    assert restarted.has_field_at(WeatherField.TEMPERATURE_2M, None)
    assert restarted.health_status()["last_update_error"] == (
        "simulated upstream outage"
    )


def test_worker_with_legacy_snapshot_resurrects_ecmwf_on_next_state(
    tmp_path,
    monkeypatch,
):
    _small_grid(monkeypatch)
    worker = ECMWFGrid(cache_dir=tmp_path, cleanup_tmp=False)
    stores = {"frame_store": object(), "ecmwf_grid": worker}
    _drop_absent_stores(stores, refreshed=["frame_store"])
    assert stores["ecmwf_grid"] is worker
    assert worker.reference_time is None

    producer = ECMWFGrid(cache_dir=tmp_path)
    run = "2026-08-06T00:00:00Z"
    assert producer._publish_prepared(
        _prepare_run(producer, run, 15.0),
        reference_time=run,
        last_modified_time=run,
    )
    dump_state({"ecmwf_grid": producer}, tmp_path)
    payload = load_state(tmp_path)
    assert payload is not None
    legacy_payload = {"stores": {"frame_store": {}}}

    assert apply_state(payload, stores, prev_payload=legacy_payload) == [
        "ecmwf_grid"
    ]
    assert worker.reference_time == run
    assert worker.available_timestamps() == [1_700_000_000, 1_700_003_600]


def test_weather_fields_can_be_disabled_without_disabling_ifs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(ifs_module.settings, "weather_fields_enabled", False)
    grid = ECMWFGrid(cache_dir=tmp_path)

    assert grid.available_fields() == frozenset({WeatherField.PRECIPITATION})
    assert grid.available_timestamps() == []


def test_render_worker_never_cleans_pipeline_temporary_files(
    tmp_path,
    monkeypatch,
):
    _small_grid(monkeypatch)
    pending = (
        Path(tmp_path)
        / "ecmwf_ifs"
        / "runs"
        / "pending-run"
        / "field.dat.tmp"
    )
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b"pipeline-write-in-progress")

    ECMWFGrid(cache_dir=tmp_path, cleanup_tmp=False)

    assert pending.exists()


async def test_single_mode_fetch_publication_invalidates_only_nwp_cache():
    class _PublishingGrid:
        model_version = "run-a"

        async def fetch(self):
            self.model_version = "run-b"
            return True

    cache = TileCache(max_mb=1)
    cache.put(("weather", "old"), b"weather")
    cache.put((1_700_000_000, "radar"), b"radar")
    cache.put(("sat", "keep"), b"satellite")
    cache.put(("cov", "keep"), b"coverage")
    fetcher = RadarFetcher.__new__(RadarFetcher)
    fetcher._cache = cache
    fetcher._nwp_contributions = [
        NWPContribution(_PublishingGrid(), 1000, "ECMWF IFS")
    ]
    fetcher._satellite_contributions = []
    fetcher._satellite_tasks = {}

    await fetcher._fetch_auxiliary_grids()

    assert cache.get(("weather", "old")) is None
    assert cache.get((1_700_000_000, "radar")) is None
    assert cache.get(("sat", "keep")) == b"satellite"
    assert cache.get(("cov", "keep")) == b"coverage"
