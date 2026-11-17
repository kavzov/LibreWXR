# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Delayed source publications must not turn repeated slots into zero motion."""
import numpy as np
import pytest

from librewxr.config import settings
from librewxr.data.nowcast import NowcastGenerator, NowcastStore
from librewxr.data.store import RadarFrame


class Store:
    def __init__(self, frames):
        self.frames = {f.timestamp: f for f in frames}

    async def get_timestamps(self):
        return sorted(self.frames)

    async def get_frame(self, ts):
        return self.frames.get(ts)


def scan(x):
    y, xx = np.ogrid[:120, :240]
    return np.where((xx-x)**2 + (y-60)**2 <= 400, 150, 0).astype(np.uint8)


def frame(slot, observed, x):
    return RadarFrame(slot, {'TEST': scan(x)}, {'TEST': observed})


async def pairs(frames):
    store = Store(frames)
    gen = NowcastGenerator(store, NowcastStore(), nowcast_contributions=[])
    previous = dict(frames[-2].regions)
    scales = await gen._observation_pairs(frames[-1], previous,
                                         await store.get_timestamps(), 300)
    return previous, scales


async def test_repeated_publication_uses_previous_distinct_scan():
    frames = [frame(1000, 1000, 100), frame(1300, 1300, 104),
              frame(1600, 1300, 104), frame(1900, 1300, 104)]
    previous, scales = await pairs(frames)
    assert previous['TEST'] is frames[0].regions['TEST']
    assert scales == {'TEST': 1.0}  # 300 s of real motion, not 900 s of slots


async def test_gap_is_scaled_by_source_time_not_slot_time():
    frames = [frame(1000, 1000, 100), frame(1300, 1000, 100),
              frame(1600, 1600, 108), frame(1900, 1600, 108)]
    previous, scales = await pairs(frames)
    assert previous['TEST'] is frames[1].regions['TEST']
    assert scales == {'TEST': 0.5}


@pytest.mark.parametrize('frames', [
    [frame(1000, None, 100), frame(1300, 1300, 104)],
    [frame(1000, 1000, 100), frame(1300, None, 104)],
    [frame(1000, 1000, 100), frame(1300, 1000, 100)],
    [frame(1000, 1000, 100), frame(1900, 1900, 104)],
    [frame(1000, 1000, 100), frame(2200, 1300, 104)],
    [frame(1000, 1300, 100), frame(1300, 1000, 104)],
])
async def test_unknown_repeated_stale_or_unbounded_pair_is_not_used(frames):
    previous, scales = await pairs(frames)
    assert previous == {}
    assert scales == {}


async def test_stationary_distinct_scans_remain_stationary():
    frames = [frame(1000, 1000, 100), frame(1300, 1300, 100)]
    previous, scales = await pairs(frames)
    assert np.array_equal(previous['TEST'], frames[-1].regions['TEST'])
    assert scales == {}


async def test_generate_moves_forecast_despite_repeated_latest_slots(monkeypatch):
    for key, value in {'fetch_interval': 300, 'nowcast_enabled': True,
                       'nowcast_frames': 2, 'nowcast_blend_mode': 'radar',
                       'radar_animation_substeps': 2, 'arrow_flow_enabled': False,
                       'nowcast_coarsen_enabled': False}.items():
        monkeypatch.setattr(settings, key, value)
    frames = [frame(1000, 1000, 100), frame(1300, 1300, 104),
              frame(1600, 1300, 104)]
    forecast = NowcastStore()
    gen = NowcastGenerator(Store(frames), forecast, nowcast_contributions=[])
    await gen.generate()
    for ts, expected_x in [(1900, 108), (2200, 112)]:
        result, _ = await forecast.get_frame(ts)
        yy, xx = np.nonzero(result.regions['TEST'] > 75)
        assert xx.mean() == pytest.approx(expected_x, abs=0.8)
    # Distinct source pairing must not smear the repeated observed interval.
    result = await forecast.get_animation_frame(1450)
    assert result is not None
    assert np.array_equal(result.regions['TEST'], frames[-1].regions['TEST'])
