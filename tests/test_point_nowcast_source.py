# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Point-nowcast ``source`` block: which region answered and what it observes."""

import asyncio
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from librewxr.api import routes
from librewxr.data.nowcast import NowcastFrame
from librewxr.data.point_nowcast import build_point_nowcast
from librewxr.data.regions import REGIONS, RegionDef
from librewxr.data.store import RadarFrame
from librewxr.mcp.sampling import dbz_to_rate_mmh

pytestmark = pytest.mark.api


def _encoded(dbz: float) -> np.uint8:
    return np.uint8(round((dbz + 32.0) * 2.0))


def _enable_test_region(monkeypatch) -> RegionDef:
    """A 0.01° radar region at the origin, covering every point it contains."""
    region = RegionDef(
        name="TEST_POINT",
        west=0.0,
        east=0.1,
        south=0.0,
        north=0.1,
        pixel_size=0.01,
        group="TEST",
    )
    monkeypatch.setitem(REGIONS, region.name, region)
    monkeypatch.setattr(
        "librewxr.mcp.sampling.sample_coverage",
        lambda _name, lat, _lon: np.ones(lat.shape, dtype=bool),
    )
    return region


class _MemoryFrameStore:
    """In-memory stand-in for FrameStore (no memmap files for large grids)."""

    def __init__(self, frames: list[RadarFrame]):
        self._frames = {frame.timestamp: frame for frame in frames}

    async def get_timestamps(self) -> list[int]:
        return sorted(self._frames)

    async def get_frame(self, timestamp: int) -> RadarFrame | None:
        return self._frames.get(timestamp)


class _MemoryNowcastStore:
    def __init__(self, frames: list[NowcastFrame]):
        self._frames = {frame.timestamp: frame for frame in frames}

    async def get_timestamps(self) -> list[int]:
        return sorted(self._frames)

    async def get_frame(self, timestamp: int):
        frame = self._frames[timestamp]
        return frame, frame.blend_weight


def _rrqpe_point_stores(lat: float, lon: float, latest: int):
    region = REGIONS["RRQPE"]
    row = int(round((region.north - lat) / region._ps_y))
    col = int(round((lon - region.west) / region.pixel_size))
    current = np.zeros((region.height, region.width), dtype=np.uint8)
    current[row, col] = _encoded(35.0)
    forecast = np.zeros_like(current)
    forecast[row, col] = _encoded(30.0)
    observed = _MemoryFrameStore([RadarFrame(latest, {"RRQPE": current})])
    nowcast = _MemoryNowcastStore([
        NowcastFrame(latest + 300, {"RRQPE": forecast}, blend_weight=1.0),
    ])
    return observed, nowcast


def test_rrqpe_region_declares_satellite_source() -> None:
    from librewxr.sources.world.rrqpe.grid import RRQPE_LAG_SECONDS

    region = REGIONS["RRQPE"]
    assert region.observation_type == "satellite"
    assert region.nominal_lag_seconds == RRQPE_LAG_SECONDS
    assert region.attribution == (
        "Precipitation data from NOAA Enterprise Rain Rate (RRQPE)"
    )
    # Every Doppler composite keeps the radar default.
    assert {
        r.observation_type for name, r in REGIONS.items() if name != "RRQPE"
    } == {"radar"}


def test_point_nowcast_labels_rrqpe_as_satellite(monkeypatch) -> None:
    monkeypatch.setattr(
        "librewxr.mcp.sampling.sample_coverage",
        lambda _name, lat, _lon: np.ones(lat.shape, dtype=bool),
    )
    monkeypatch.setattr("librewxr.config.settings.rrqpe_dbz_offset", -7.0)
    lat, lon = 28.6139, 77.2090  # New Delhi: no Doppler region covers it
    latest = int(time.time() // 300) * 300
    observed, nowcast = _rrqpe_point_stores(lat, lon, latest)

    result = asyncio.run(build_point_nowcast(
        frame_store=observed,
        nowcast_store=nowcast,
        enabled_regions=["RRQPE"],
        lat=lat,
        lon=lon,
        radius_km=2.0,
        past_minutes=30,
        future_minutes=60,
        noise_floor_dbz=10.0,
        fetch_interval=300,
    ))

    assert result["source"] == {
        "region": "RRQPE",
        "type": "satellite",
        # 0.04° rows are ~4.45 km; columns shrink with cos(latitude).
        "grid_spacing_km": pytest.approx(4.45, abs=0.01),
        "nominal_lag_seconds": 1800,
        "attribution": "Precipitation data from NOAA Enterprise Rain Rate (RRQPE)",
    }
    assert [frame["period"] for frame in result["frames"]] == ["observed", "forecast"]
    assert all(frame["region"] == "RRQPE" for frame in result["frames"])
    assert all(frame["coverage"] == "in_range" for frame in result["frames"])
    # Stored values carry the display offset; the API reports true dBZ.
    assert result["frames"][0]["max_dbz"] == 42.0
    assert result["frames"][1]["max_dbz"] == 37.0
    assert result["frames"][0]["max_rate_mmh"] == pytest.approx(
        dbz_to_rate_mmh(42.0)
    )

    from librewxr.api.models import RadarPointNowcastResponse

    model = RadarPointNowcastResponse(**result)
    assert model.source is not None and model.source.type == "satellite"


def test_point_nowcast_prefers_radar_over_rrqpe(monkeypatch) -> None:
    region = _enable_test_region(monkeypatch)
    latest = int(time.time() // 300) * 300
    radar_frame = np.zeros((10, 10), dtype=np.uint8)
    rrqpe_frame = np.zeros(
        (REGIONS["RRQPE"].height, REGIONS["RRQPE"].width), dtype=np.uint8,
    )
    observed = _MemoryFrameStore([
        RadarFrame(latest, {region.name: radar_frame, "RRQPE": rrqpe_frame}),
    ])

    result = asyncio.run(build_point_nowcast(
        frame_store=observed,
        nowcast_store=None,
        enabled_regions=["RRQPE", region.name],
        lat=0.05,
        lon=0.05,
        radius_km=1.2,
        past_minutes=30,
        future_minutes=60,
        noise_floor_dbz=10.0,
        fetch_interval=300,
    ))

    assert result["source"] == {
        "region": region.name,
        "type": "radar",
        "grid_spacing_km": pytest.approx(1.11, abs=0.01),
        "nominal_lag_seconds": None,
        "attribution": None,
    }
    assert result["frames"][0]["region"] == region.name


def test_point_nowcast_endpoint_reports_source(monkeypatch) -> None:
    monkeypatch.setattr(
        "librewxr.mcp.sampling.sample_coverage",
        lambda _name, lat, _lon: np.ones(lat.shape, dtype=bool),
    )
    latest = int(time.time() // 300) * 300
    observed, nowcast = _rrqpe_point_stores(6.5244, 3.3792, latest)  # Lagos
    monkeypatch.setattr(routes, "frame_store", observed)
    monkeypatch.setattr(routes, "nowcast_store", nowcast)
    monkeypatch.setattr(routes, "enabled_regions", ["RRQPE"])

    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        covered = client.get(
            "/v2/radar/point-nowcast.json",
            params={"lat": 6.5244, "lon": 3.3792},
        )
        # Poleward of the 60S-70N band nothing covers the point.
        uncovered = client.get(
            "/v2/radar/point-nowcast.json",
            params={"lat": 80.0, "lon": 3.3792},
        )

    assert covered.status_code == 200
    source = covered.json()["source"]
    assert source["region"] == "RRQPE"
    assert source["type"] == "satellite"
    assert source["nominal_lag_seconds"] == 1800
    assert uncovered.status_code == 200
    assert uncovered.json()["source"] is None
    assert all(
        frame["coverage"] == "out_of_range"
        for frame in uncovered.json()["frames"]
    )


def test_point_nowcast_source_null_when_resolved_region_has_no_data(monkeypatch) -> None:
    """A bbox-resolved region with no frames (provider disabled) is not a source."""
    monkeypatch.setattr(
        "librewxr.mcp.sampling.sample_coverage",
        lambda _name, lat, _lon: np.ones(lat.shape, dtype=bool),
    )
    latest = int(time.time() // 300) * 300
    # RRQPE is in the enabled list (as under ENABLED_REGIONS=ALL) but the
    # frames carry no RRQPE array, as when LIBREWXR_RRQPE_ENABLED=false.
    observed = _MemoryFrameStore([
        RadarFrame(latest, {"SVCOMP": np.zeros((2, 2), dtype=np.uint8)}),
    ])

    result = asyncio.run(build_point_nowcast(
        frame_store=observed,
        nowcast_store=None,
        enabled_regions=["RRQPE"],
        lat=28.6139,
        lon=77.2090,
        radius_km=2.0,
        past_minutes=30,
        future_minutes=60,
        noise_floor_dbz=10.0,
        fetch_interval=300,
    ))

    assert result["source"] is None
    assert result["frames"][0]["region"] == "RRQPE"
    assert result["frames"][0]["coverage"] == "out_of_range"


def test_point_nowcast_falls_through_dataless_radar_to_rrqpe(monkeypatch) -> None:
    """Manila case: a finer radar region with no frames must not shadow RRQPE."""
    region = _enable_test_region(monkeypatch)
    latest = int(time.time() // 300) * 300
    rrqpe = REGIONS["RRQPE"]
    rrqpe_frame = np.zeros((rrqpe.height, rrqpe.width), dtype=np.uint8)
    observed = _MemoryFrameStore([RadarFrame(latest, {"RRQPE": rrqpe_frame})])

    result = asyncio.run(build_point_nowcast(
        frame_store=observed,
        nowcast_store=None,
        enabled_regions=[region.name, "RRQPE"],
        lat=0.05,
        lon=0.05,
        radius_km=2.0,
        past_minutes=30,
        future_minutes=60,
        noise_floor_dbz=10.0,
        fetch_interval=300,
    ))

    assert result["source"]["region"] == "RRQPE"
    assert result["source"]["type"] == "satellite"
    assert result["frames"][0]["region"] == "RRQPE"
    assert result["frames"][0]["coverage"] == "in_range"

