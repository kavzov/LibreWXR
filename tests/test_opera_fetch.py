# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""ORD endpoint failover and source time, without network requests."""
import io
import subprocess
import sys
from datetime import datetime, timezone

import h5py
import httpx
import numpy as np
import pytest

from librewxr.sources.regional.europe.radar.opera.source import OperaSource

pytestmark = pytest.mark.sources
TS = 1794890400  # 2026-11-17 04:40 UTC


def test_opera_discovery_when_source_protocol_is_imported_first():
    subprocess.run([
        sys.executable, "-c",
        "from librewxr.sources._base import RadarObservation; "
        "from librewxr.data.regions import REGIONS; assert 'OPERA' in REGIONS",
    ], check=True, capture_output=True, text=True)


def product(timestamp=TS):
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        dt = datetime.fromtimestamp(timestamp, timezone.utc)
        what = f.create_group("what")
        what.attrs["date"] = dt.strftime("%Y%m%d").encode()
        what.attrs["time"] = dt.strftime("%H%M%S").encode()
        d = f.create_group("dataset1/data1")
        d.create_dataset("data", data=np.full((2, 2), 30.0))
        w = d.create_group("what")
        for key, value in {"gain": 1., "offset": 0., "nodata": -9999000., "undetect": -8888000.}.items():
            w.attrs[key] = value
    return buf.getvalue()


@pytest.mark.parametrize("failure", [404, 503, "transport", "invalid", "misdated"])
async def test_backup_serves_current_product_before_older_primary(monkeypatch, failure):
    calls = []
    async def get(client, url, **kwargs):
        calls.append(url)
        if "cloudferro" in url:
            if failure == "transport":
                return None
            if failure == "invalid":
                return httpx.Response(200, content=b"broken hdf5")
            if failure == "misdated":
                return httpx.Response(200, content=product(TS - 300))
            return httpx.Response(failure)
        return httpx.Response(200, content=product())
    monkeypatch.setattr("librewxr.sources.regional.europe.radar.opera.source.retry_get", get)
    source = OperaSource()
    try:
        result = await source._fetch_hdf5(TS)
        assert result.observed_at == TS
        assert np.all(result.data == 124)
        assert len(calls) == 2
        assert "openradar-24h-backup/2026/11/17/" in calls[1]
        assert all("T0440@" in url for url in calls)
    finally:
        await source.close()


async def test_primary_success_does_not_request_backup(monkeypatch):
    calls = []
    async def get(client, url, **kwargs):
        calls.append(url)
        return httpx.Response(200, content=product())
    monkeypatch.setattr("librewxr.sources.regional.europe.radar.opera.source.retry_get", get)
    source = OperaSource()
    try:
        assert (await source._fetch_hdf5(TS)).observed_at == TS
        assert len(calls) == 1
    finally:
        await source.close()


@pytest.mark.parametrize("available", [True, False])
async def test_older_product_keeps_source_time_and_attempts_are_bounded(monkeypatch, available):
    calls = []
    async def get(client, url, **kwargs):
        calls.append(url)
        if available and "backup" in url and "T0435@" in url:
            return httpx.Response(200, content=product(TS - 300))
        return httpx.Response(404)
    monkeypatch.setattr("librewxr.sources.regional.europe.radar.opera.source.retry_get", get)
    source = OperaSource()
    try:
        result = await source._fetch_hdf5(TS)
        if available:
            assert result.observed_at == TS - 300
            assert len(calls) == 4
        else:
            assert result is None
            assert len(calls) == 8
    finally:
        await source.close()
