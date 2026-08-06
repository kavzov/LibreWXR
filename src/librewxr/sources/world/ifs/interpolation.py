# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Optical-flow temporal interpolation for ECMWF IFS hourly grids.

Adapter around the shared ``nwp_interpolation`` helper that the
regional NWP sources also use. Only precipitation and its categorical snow
mask are interpolated; native physical weather fields remain on their actual
valid times and are interpolated after spatial sampling by ``ECMWFGrid``.
"""
from __future__ import annotations

import logging

import numpy as np

from librewxr.data.nwp_interpolation import interpolate_run
from librewxr.data.weather_fields import WeatherField
from librewxr.sources.world.ifs.models import WeatherFrame

logger = logging.getLogger(__name__)


def interpolate_timesteps(
    timesteps: dict[int, WeatherFrame],
    interval_seconds: int = 600,
) -> dict[int, WeatherFrame]:
    """Create sub-hourly IFS frames by optical-flow interpolation.

    Adapter shim: extracts precipitation and snow into parallel dicts, calls
    the shared interpolator, and re-packs without touching other fields.

    Args:
        timesteps: Native frames. Frames without precipitation are preserved
            but do not participate in optical-flow interpolation.
        interval_seconds: Target interval between frames (default 600
            = 10 min, matching the radar cadence).

    Returns:
        New dict containing both original and interpolated timesteps.
        The warp's internal flow field is no longer surfaced — the
        hybrid arrow path builds a composite NWP flow raster instead
        (see ``NowcastGenerator._compute_nwp_flow_sync``), so IFS no
        longer needs to carry its own flow for arrow rendering.
    """
    precip_frames = {
        ts: frame
        for ts, frame in timesteps.items()
        if frame.has_field(WeatherField.PRECIPITATION)
    }
    if len(precip_frames) < 2:
        return dict(timesteps)

    precip_by_ts = {
        ts: frame.field(WeatherField.PRECIPITATION)
        for ts, frame in precip_frames.items()
    }
    snow_by_ts = {
        ts: frame.snow_mask
        for ts, frame in precip_frames.items()
        if frame.snow_mask is not None
    }

    aug_precip, aug_snow, _last_flow = interpolate_run(
        precip_by_ts,
        snow_by_ts if len(snow_by_ts) == len(precip_by_ts) else None,
        target_interval_seconds=interval_seconds,
        log_label="ECMWF interpolation",
    )

    result = dict(timesteps)
    for ts in aug_precip:
        original = timesteps.get(ts)
        fields = dict(original.fields) if original is not None else {}
        fields[WeatherField.PRECIPITATION] = aug_precip[ts]
        snow = (
            aug_snow[ts]
            if aug_snow is not None
            else (original.snow_mask if original is not None else None)
        )
        # IFS expects snow to be bool; the shared helper preserves bool
        # dtype if input was bool, so this is just defensive.
        if snow is not None and snow.dtype != np.bool_:
            snow = snow.astype(bool)
        result[ts] = WeatherFrame(timestamp=ts, fields=fields, snow_mask=snow)
    return result
