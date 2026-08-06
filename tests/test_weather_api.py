# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""API tests for global scalar weather metadata and tiles."""

from __future__ import annotations

import io
import asyncio
import time
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from librewxr.api import routes
from librewxr.colors.weather_palettes import WEATHER_PALETTES
from librewxr.config import settings
from librewxr.data.nwp_source import NWPChain
from librewxr.data.store import FrameStore
from librewxr.data.weather_fields import (
    WeatherField,
    encode_field,
    relative_humidity_from_temperature_dewpoint,
)
from librewxr.data.weather_sampling import clear_sampling_plan_cache
from librewxr.sources.world.ifs import grid as ifs_module
from librewxr.sources.world.ifs.grid import ECMWFGrid
from librewxr.sources.world.ifs.models import WeatherFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.cache import CachedRender
from librewxr.tiles.weather_renderer import colorize_weather_values

pytestmark = pytest.mark.api


@pytest.fixture
def weather_api(tmp_path, monkeypatch):
    monkeypatch.setattr(ifs_module, "PIXEL_SIZE", 90.0)
    monkeypatch.setattr(ifs_module, "GRID_WIDTH", 4)
    monkeypatch.setattr(ifs_module, "GRID_HEIGHT", 3)
    monkeypatch.setattr(ifs_module, "GRID_SHAPE", (3, 4))
    clear_sampling_plan_cache()

    t0 = int(time.time() // 3600) * 3600
    t1 = t0 + 3600

    def encoded_fields(
        temperature: float,
        dewpoint: float,
        pressure: float,
    ) -> dict[WeatherField, np.ndarray]:
        shape = (3, 4)
        return {
            WeatherField.PRECIPITATION: np.full(shape, 96, dtype=np.uint8),
            WeatherField.TEMPERATURE_2M: encode_field(
                WeatherField.TEMPERATURE_2M, np.full(shape, temperature)
            ),
            WeatherField.DEWPOINT_2M: encode_field(
                WeatherField.DEWPOINT_2M, np.full(shape, dewpoint)
            ),
            WeatherField.PRESSURE_MSL: encode_field(
                WeatherField.PRESSURE_MSL, np.full(shape, pressure)
            ),
            WeatherField.WIND_U_10M: encode_field(
                WeatherField.WIND_U_10M, np.full(shape, 3.0)
            ),
            WeatherField.WIND_V_10M: encode_field(
                WeatherField.WIND_V_10M, np.full(shape, 4.0)
            ),
        }

    grid = ECMWFGrid(cache_dir=tmp_path)
    grid._timesteps[t0] = WeatherFrame(t0, encoded_fields(12.0, 7.0, 1000.0))
    grid._timesteps[t1] = WeatherFrame(t1, encoded_fields(22.0, 17.0, 1010.0))
    grid._reference_time = datetime.fromtimestamp(t0, timezone.utc).isoformat()
    chain = NWPChain([grid])
    cache = TileCache(max_mb=10)

    previous = (
        routes.ecmwf_grid,
        routes.nwp_chain,
        routes.tile_cache,
        routes.frame_store,
        routes.nowcast_store,
        routes.satellite_grids,
    )
    routes.ecmwf_grid = grid
    routes.nwp_chain = chain
    routes.tile_cache = cache
    routes.frame_store = FrameStore(max_frames=1)
    routes.nowcast_store = None
    routes.satellite_grids = {}
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield {
            "client": client,
            "grid": grid,
            "chain": chain,
            "cache": cache,
            "t0": t0,
            "t1": t1,
            "midpoint": t0 + 1800,
        }
    (
        routes.ecmwf_grid,
        routes.nwp_chain,
        routes.tile_cache,
        routes.frame_store,
        routes.nowcast_store,
        routes.satellite_grids,
    ) = previous
    clear_sampling_plan_cache()


def _url(context, field, palette, *, ext="png", size=256, z=0, x=0, y=0):
    return (
        f"/v2/weather/{field}/{context['midpoint']}/{size}/"
        f"{z}/{x}/{y}/{palette}.{ext}"
    )


def test_metadata_schema_and_legend(weather_api):
    response = weather_api["client"].get("/v2/weather/metadata.json")

    assert response.status_code == 200
    body = response.json()
    assert body["active_model_run"] == weather_api["grid"].reference_time
    assert isinstance(body["generated"], int)
    assert body["stale"] is False
    assert "ECMWF" in body["attribution"]
    assert body["available_timestamps"] == [weather_api["t0"], weather_api["t1"]]
    assert body["default_timestamp"] in body["available_timestamps"]
    assert body["sizes"] == [256, 512]
    assert body["formats"] == ["png", "webp"]
    assert body["min_zoom"] == 0
    assert body["max_zoom"] == settings.max_zoom
    assert "{field}" in body["tile_url_template"]
    assert body["palette_ids"] == list(WEATHER_PALETTES)

    fields = {field["id"]: field for field in body["fields"]}
    assert set(fields) == {
        "temperature_2m",
        "dewpoint_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "wind_speed_10m",
    }
    assert fields["temperature_2m"]["unit"] == "°C"
    assert fields["pressure_msl"]["unit"] == "hPa"
    palettes = {palette["id"]: palette for palette in body["palettes"]}
    assert palettes["temperature"]["stops"]
    assert palettes["temperature"]["unit"] == "°C"
    assert palettes["humidity"]["minimum"] == 0.0
    assert palettes["humidity"]["maximum"] == 100.0
    assert response.headers["cache-control"] == "public, max-age=60"


def test_rain_viewer_metadata_has_no_weather_extension(weather_api):
    response = weather_api["client"].get("/public/weather-maps.json")

    assert response.status_code == 200
    assert "weather" not in response.json()


@pytest.mark.parametrize(
    ("path", "status", "detail"),
    [
        ("/v2/weather/unknown/{ts}/256/0/0/0/temperature.png", 404, "Unknown"),
        ("/v2/weather/temperature_2m/{before}/256/0/0/0/temperature.png", 404, "outside"),
        ("/v2/weather/temperature_2m/{ts}/256/0/0/0/humidity.png", 400, "palette"),
        ("/v2/weather/temperature_2m/{ts}/128/0/0/0/temperature.png", 400, "size"),
        ("/v2/weather/temperature_2m/{ts}/256/{zoom}/0/0/temperature.png", 400, "Zoom"),
        ("/v2/weather/temperature_2m/{ts}/256/1/2/0/temperature.png", 400, "coordinates"),
        ("/v2/weather/temperature_2m/{ts}/256/0/0/0/temperature.gif", 422, None),
    ],
)
def test_weather_tile_validation(weather_api, path, status, detail):
    path = path.format(
        ts=weather_api["midpoint"],
        before=weather_api["t0"] - 1,
        zoom=settings.max_zoom + 1,
    )
    response = weather_api["client"].get(path)

    assert response.status_code == status
    if detail is not None:
        assert detail.lower() in response.json()["detail"].lower()


def test_png_and_webp_content_types_and_signatures(weather_api):
    png = weather_api["client"].get(
        _url(weather_api, "temperature_2m", "temperature")
    )
    webp = weather_api["client"].get(
        _url(weather_api, "wind_speed_10m", "wind_speed", ext="webp")
    )

    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert webp.status_code == 200
    assert webp.headers["content-type"] == "image/webp"
    assert webp.content[:4] == b"RIFF"
    assert webp.content[8:12] == b"WEBP"

    large = weather_api["client"].get(
        _url(weather_api, "temperature_2m", "temperature", size=512)
    )
    assert large.status_code == 200
    assert Image.open(io.BytesIO(large.content)).size == (512, 512)


def test_weather_tile_etag_and_conditional_request(weather_api):
    url = _url(weather_api, "pressure_msl", "pressure")
    first = weather_api["client"].get(url)
    second = weather_api["client"].get(
        url, headers={"If-None-Match": first.headers["etag"]}
    )

    assert first.status_code == 200
    assert first.headers["etag"].startswith('"')
    assert "public" in first.headers["cache-control"]
    assert second.status_code == 304
    assert second.content == b""
    assert second.headers["etag"] == first.headers["etag"]


def test_weather_tile_cache_hit_avoids_rerender(weather_api, monkeypatch):
    weather_api["cache"].clear()
    calls = 0
    original = routes.render_scalar_weather_tile

    def counted_render(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(routes, "render_scalar_weather_tile", counted_render)
    url = _url(weather_api, "dewpoint_2m", "dewpoint", z=1, x=0, y=0)
    first = weather_api["client"].get(url)
    second = weather_api["client"].get(url)

    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert calls == 1
    entries = weather_api["cache"].entries()
    assert len(entries) == 1
    key = entries[0][0]
    assert key[0] == "weather"
    assert key[1] == "dewpoint_2m"
    assert weather_api["grid"].model_version in key


@pytest.mark.parametrize(
    ("field", "palette_id", "expected"),
    [
        ("temperature_2m", "temperature", 17.0),
        ("dewpoint_2m", "dewpoint", 12.0),
        (
            "relative_humidity_2m",
            "humidity",
            float(relative_humidity_from_temperature_dewpoint(17.0, 12.0)),
        ),
        ("pressure_msl", "pressure", 1005.0),
        ("wind_speed_10m", "wind_speed", 5.0),
    ],
)
def test_physical_and_derived_field_samples_have_no_transparent_holes(
    weather_api,
    field,
    palette_id,
    expected,
):
    response = weather_api["client"].get(_url(weather_api, field, palette_id))

    assert response.status_code == 200
    rgba = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGBA"))
    expected_rgba = colorize_weather_values(
        np.asarray([[expected]], dtype=np.float32),
        WEATHER_PALETTES[palette_id],
    )[0, 0]
    np.testing.assert_array_equal(rgba[128, 128], expected_rgba)
    assert (rgba[..., 3] == 255).all()


def test_adjacent_global_tiles_have_no_seam_or_empty_pixels(weather_api):
    left = weather_api["client"].get(
        _url(weather_api, "pressure_msl", "pressure", z=1, x=0, y=0)
    )
    right = weather_api["client"].get(
        _url(weather_api, "pressure_msl", "pressure", z=1, x=1, y=0)
    )
    left_rgba = np.asarray(Image.open(io.BytesIO(left.content)).convert("RGBA"))
    right_rgba = np.asarray(Image.open(io.BytesIO(right.content)).convert("RGBA"))

    assert left.status_code == right.status_code == 200
    np.testing.assert_array_equal(left_rgba[:, -1], right_rgba[:, 0])
    assert (left_rgba[..., 3] == 255).all()
    assert (right_rgba[..., 3] == 255).all()


async def test_weather_singleflight_shares_one_render_and_cleans_success():
    routes._weather_tile_flights.clear()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def factory():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return CachedRender(b"tile", '"etag"')

    requests = [
        asyncio.create_task(routes._weather_tile_singleflight(("same",), factory))
        for _ in range(4)
    ]
    await started.wait()
    release.set()
    results = await asyncio.gather(*requests)
    await asyncio.sleep(0)

    assert calls == 1
    assert [result.data for result in results] == [b"tile"] * 4
    assert routes._weather_tile_flights == {}


async def test_weather_singleflight_removes_failed_and_cancelled_waiters():
    routes._weather_tile_flights.clear()
    attempts = 0

    async def failed_factory():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        await routes._weather_tile_singleflight(("failed",), failed_factory)
    await asyncio.sleep(0)
    assert ("failed",) not in routes._weather_tile_flights

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_factory():
        started.set()
        await release.wait()
        return CachedRender(b"ok", '"ok"')

    leader = asyncio.create_task(
        routes._weather_tile_singleflight(("cancel",), slow_factory)
    )
    await started.wait()
    follower = asyncio.create_task(
        routes._weather_tile_singleflight(("cancel",), slow_factory)
    )
    follower.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower
    release.set()
    assert (await leader).data == b"ok"
    await asyncio.sleep(0)
    assert ("cancel",) not in routes._weather_tile_flights
