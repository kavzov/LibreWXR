# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

import asyncio
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from librewxr.api import routes
from librewxr.data.nowcast import NowcastFrame, NowcastStore
from librewxr.data.point_nowcast import build_point_nowcast, sample_neighborhood
from librewxr.data.regions import REGIONS, RegionDef
from librewxr.data.store import FrameStore, RadarFrame

pytestmark = pytest.mark.api


def _region() -> RegionDef:
    return RegionDef(
        name="TEST_POINT",
        west=0.0,
        east=0.1,
        south=0.0,
        north=0.1,
        pixel_size=0.01,
        group="TEST",
    )


def _encoded(dbz: float) -> np.uint8:
    return np.uint8(round((dbz + 32.0) * 2.0))


def _stores(latest: int) -> tuple[FrameStore, NowcastStore]:
    observed = FrameStore(max_frames=4)
    nowcast = NowcastStore()

    old = np.zeros((10, 10), dtype=np.uint8)
    old[5, 5] = _encoded(20.0)
    current = np.zeros((10, 10), dtype=np.uint8)
    forecast_5 = np.zeros((10, 10), dtype=np.uint8)
    forecast_5[5, 5] = _encoded(25.0)
    forecast_10 = np.zeros((10, 10), dtype=np.uint8)
    forecast_10[5, 5] = _encoded(30.0)

    asyncio.run(observed.add_frame(RadarFrame(latest - 600, {"TEST_POINT": old})))
    asyncio.run(observed.add_frame(RadarFrame(latest, {"TEST_POINT": current})))
    asyncio.run(nowcast.replace_all([
        NowcastFrame(latest + 300, {"TEST_POINT": forecast_5}, blend_weight=0.8),
        NowcastFrame(latest + 600, {"TEST_POINT": forecast_10}, blend_weight=0.6),
    ]))
    return observed, nowcast


def _enable_test_region(monkeypatch) -> RegionDef:
    region = _region()
    monkeypatch.setitem(REGIONS, region.name, region)
    monkeypatch.setattr(
        "librewxr.mcp.sampling.sample_coverage",
        lambda _name, lat, _lon: np.ones(lat.shape, dtype=bool),
    )
    return region


def test_sample_neighborhood_uses_noise_floor() -> None:
    region = _region()
    frame = np.zeros((10, 10), dtype=np.uint8)
    frame[5, 5] = _encoded(20.0)
    frame[5, 6] = _encoded(5.0)

    sample = sample_neighborhood(
        region,
        lat=0.05,
        lon=0.05,
        radius_km=1.2,
        frame_array=frame,
        noise_floor_dbz=10.0,
    )

    assert sample.coverage == "in_range"
    assert sample.sample_count == 5
    assert sample.wet_pixel_count == 1
    assert sample.wet_fraction == pytest.approx(0.2)
    assert sample.max_dbz == 20.0
    assert sample.max_rate_mmh is not None
    assert sample.max_rate_mmh > 0.0


def test_sample_neighborhood_outside_array() -> None:
    sample = sample_neighborhood(
        _region(),
        lat=1.0,
        lon=1.0,
        radius_km=2.0,
        frame_array=np.zeros((10, 10), dtype=np.uint8),
        noise_floor_dbz=10.0,
    )

    assert sample.coverage == "out_of_range"
    assert sample.sample_count == 0
    assert sample.wet_fraction is None


def test_build_point_nowcast_returns_observed_and_forecast(monkeypatch) -> None:
    _enable_test_region(monkeypatch)
    latest = int(time.time() // 300) * 300
    observed, nowcast = _stores(latest)

    result = asyncio.run(build_point_nowcast(
        frame_store=observed,
        nowcast_store=nowcast,
        enabled_regions=["TEST_POINT"],
        lat=0.05,
        lon=0.05,
        radius_km=1.2,
        past_minutes=30,
        future_minutes=60,
        noise_floor_dbz=10.0,
        fetch_interval=300,
    ))

    assert result["latest_observation_time"] == latest
    assert result["history_minutes_available"] == 10
    assert result["forecast_minutes_available"] == 10
    assert result["stale"] is False
    assert [frame["minutes_offset"] for frame in result["frames"]] == [-10, 0, 5, 10]
    assert [frame["period"] for frame in result["frames"]] == [
        "observed", "observed", "forecast", "forecast",
    ]
    assert result["frames"][1]["wet_fraction"] == 0.0
    assert result["frames"][2]["wet_fraction"] > 0.0
    assert result["frames"][2]["blend_weight"] == 0.8

    bounded = asyncio.run(build_point_nowcast(
        frame_store=observed,
        nowcast_store=nowcast,
        enabled_regions=["TEST_POINT"],
        lat=0.05,
        lon=0.05,
        radius_km=1.2,
        past_minutes=5,
        future_minutes=5,
        noise_floor_dbz=10.0,
        fetch_interval=300,
    ))
    assert [frame["minutes_offset"] for frame in bounded["frames"]] == [0, 5]


def test_build_point_nowcast_reports_stale_and_no_coverage(monkeypatch) -> None:
    region = _region()
    monkeypatch.setitem(REGIONS, region.name, region)
    monkeypatch.setattr(
        "librewxr.mcp.sampling.sample_coverage",
        lambda _name, lat, _lon: np.zeros(lat.shape, dtype=bool),
    )
    latest = int(time.time()) - 1_200
    observed, nowcast = _stores(latest)

    result = asyncio.run(build_point_nowcast(
        frame_store=observed,
        nowcast_store=nowcast,
        enabled_regions=[region.name],
        lat=0.05,
        lon=0.05,
        radius_km=2.0,
        past_minutes=30,
        future_minutes=60,
        noise_floor_dbz=10.0,
        fetch_interval=300,
    ))

    assert result["stale"] is True
    assert result["latest_age_seconds"] >= 1_200
    assert all(frame["coverage"] == "out_of_range" for frame in result["frames"])
    assert all(frame["region"] is None for frame in result["frames"])


def test_point_nowcast_endpoint_and_query_validation(monkeypatch) -> None:
    _enable_test_region(monkeypatch)
    latest = int(time.time() // 300) * 300
    observed, nowcast = _stores(latest)
    monkeypatch.setattr(routes, "frame_store", observed)
    monkeypatch.setattr(routes, "nowcast_store", nowcast)
    monkeypatch.setattr(routes, "enabled_regions", ["TEST_POINT"])

    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        response = client.get(
            "/v2/radar/point-nowcast.json",
            params={"lat": 0.05, "lon": 0.05, "radius_km": 1.2},
        )
        invalid_radius = client.get(
            "/v2/radar/point-nowcast.json",
            params={"lat": 0.05, "lon": 0.05, "radius_km": 20},
        )
        invalid_horizon = client.get(
            "/v2/radar/point-nowcast.json",
            params={"lat": 0.05, "lon": 0.05, "future_minutes": 61},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["radius_km"] == 1.2
    assert data["forecast_minutes_available"] == 10
    assert data["frames"][-1]["period"] == "forecast"
    assert response.headers["cache-control"] == "private, max-age=60"
    assert invalid_radius.status_code == 422
    assert invalid_horizon.status_code == 422


def test_point_nowcast_endpoint_without_observations(monkeypatch) -> None:
    monkeypatch.setattr(routes, "frame_store", FrameStore(max_frames=1))
    monkeypatch.setattr(routes, "nowcast_store", None)
    monkeypatch.setattr(routes, "enabled_regions", [])
    app = FastAPI()
    app.include_router(routes.router)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(
            "/v2/radar/point-nowcast.json",
            params={"lat": 54.6872, "lon": 25.2797},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Radar observations not available"
