# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Stable scalar-weather palette registry with precomputed color tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from librewxr.data.weather_fields import WeatherField

WEATHER_LUT_SIZE = 4096


@dataclass(frozen=True)
class PaletteStop:
    value: float
    color: str


def _hex_rgba(color: str, opacity: float = 1.0) -> np.ndarray:
    raw = color.removeprefix("#")
    if len(raw) not in (6, 8):
        raise ValueError(f"invalid palette color: {color}")
    channels = [int(raw[index:index + 2], 16) for index in range(0, len(raw), 2)]
    if len(channels) == 3:
        channels.append(255)
    channels[3] = round(channels[3] * opacity)
    return np.asarray(channels, dtype=np.uint8)


@dataclass(frozen=True)
class WeatherPalette:
    """Immutable palette metadata plus its import-time RGBA lookup table."""

    id: str
    display_name: str
    field: WeatherField
    unit: str
    minimum: float
    maximum: float
    stops: tuple[PaletteStop, ...]
    below_color: str
    above_color: str
    nodata_color: str = "#00000000"
    opacity: float = 1.0
    lut: np.ndarray = field(init=False, repr=False, compare=False)
    below_rgba: np.ndarray = field(init=False, repr=False, compare=False)
    above_rgba: np.ndarray = field(init=False, repr=False, compare=False)
    nodata_rgba: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.minimum < self.maximum:
            raise ValueError("palette minimum must be less than maximum")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("palette opacity must be within [0, 1]")
        if len(self.stops) < 2:
            raise ValueError("a weather palette needs at least two stops")
        values = np.asarray([stop.value for stop in self.stops], dtype=np.float32)
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("palette stops must be strictly increasing")
        colors = np.stack(
            [_hex_rgba(stop.color, self.opacity) for stop in self.stops]
        ).astype(np.float32)
        positions = np.linspace(
            self.minimum, self.maximum, WEATHER_LUT_SIZE, dtype=np.float32
        )
        lut = np.empty((WEATHER_LUT_SIZE, 4), dtype=np.uint8)
        for channel in range(4):
            lut[:, channel] = np.clip(
                np.rint(np.interp(positions, values, colors[:, channel])),
                0,
                255,
            ).astype(np.uint8)
        lut.flags.writeable = False
        below = _hex_rgba(self.below_color, self.opacity)
        above = _hex_rgba(self.above_color, self.opacity)
        nodata = _hex_rgba(self.nodata_color, self.opacity)
        for color in (below, above, nodata):
            color.flags.writeable = False
        object.__setattr__(self, "lut", lut)
        object.__setattr__(self, "below_rgba", below)
        object.__setattr__(self, "above_rgba", above)
        object.__setattr__(self, "nodata_rgba", nodata)


_PALETTES = {
    "temperature": WeatherPalette(
        id="temperature",
        display_name="Air temperature",
        field=WeatherField.TEMPERATURE_2M,
        unit="°C",
        minimum=-50.0,
        maximum=50.0,
        stops=(
            PaletteStop(-50.0, "#351a87"),
            PaletteStop(-30.0, "#2757c7"),
            PaletteStop(-15.0, "#27b9e6"),
            PaletteStop(0.0, "#d7f4f1"),
            PaletteStop(10.0, "#78c850"),
            PaletteStop(20.0, "#f4dc42"),
            PaletteStop(30.0, "#f28c28"),
            PaletteStop(40.0, "#d62929"),
            PaletteStop(50.0, "#7f143c"),
        ),
        below_color="#19094f",
        above_color="#4b0828",
    ),
    "dewpoint": WeatherPalette(
        id="dewpoint",
        display_name="Dew point",
        field=WeatherField.DEWPOINT_2M,
        unit="°C",
        minimum=-60.0,
        maximum=35.0,
        stops=(
            PaletteStop(-60.0, "#4b2a7b"),
            PaletteStop(-30.0, "#355fb5"),
            PaletteStop(-10.0, "#45b6c9"),
            PaletteStop(0.0, "#a6d96a"),
            PaletteStop(10.0, "#4daf4a"),
            PaletteStop(20.0, "#16853f"),
            PaletteStop(30.0, "#08642f"),
            PaletteStop(35.0, "#00441b"),
        ),
        below_color="#2c174d",
        above_color="#002b12",
    ),
    "humidity": WeatherPalette(
        id="humidity",
        display_name="Relative humidity",
        field=WeatherField.RELATIVE_HUMIDITY_2M,
        unit="%",
        minimum=0.0,
        maximum=100.0,
        stops=(
            PaletteStop(0.0, "#7f3b08"),
            PaletteStop(20.0, "#d88b2b"),
            PaletteStop(40.0, "#f6e8a6"),
            PaletteStop(60.0, "#b8e0d2"),
            PaletteStop(80.0, "#4fa3c4"),
            PaletteStop(100.0, "#253494"),
        ),
        below_color="#7f3b08",
        above_color="#1b2370",
    ),
    "pressure": WeatherPalette(
        id="pressure",
        display_name="Mean sea-level pressure",
        field=WeatherField.PRESSURE_MSL,
        unit="hPa",
        minimum=950.0,
        maximum=1050.0,
        stops=(
            PaletteStop(950.0, "#54278f"),
            PaletteStop(970.0, "#756bb1"),
            PaletteStop(990.0, "#9ecae1"),
            PaletteStop(1010.0, "#f7f7f7"),
            PaletteStop(1030.0, "#fdae6b"),
            PaletteStop(1050.0, "#cb181d"),
        ),
        below_color="#3f176d",
        above_color="#8f0d13",
    ),
    "wind_speed": WeatherPalette(
        id="wind_speed",
        display_name="Wind speed at 10 m",
        field=WeatherField.WIND_SPEED_10M,
        unit="m/s",
        minimum=0.0,
        maximum=50.0,
        stops=(
            PaletteStop(0.0, "#f7fbff"),
            PaletteStop(5.0, "#c6dbef"),
            PaletteStop(10.0, "#6baed6"),
            PaletteStop(20.0, "#2171b5"),
            PaletteStop(30.0, "#6a51a3"),
            PaletteStop(40.0, "#ce1256"),
            PaletteStop(50.0, "#7a0177"),
        ),
        below_color="#f7fbff",
        above_color="#49006a",
    ),
}

WEATHER_PALETTES: Mapping[str, WeatherPalette] = MappingProxyType(_PALETTES)

PUBLIC_WEATHER_FIELDS: Mapping[str, WeatherField] = MappingProxyType(
    {
        "temperature_2m": WeatherField.TEMPERATURE_2M,
        "dewpoint_2m": WeatherField.DEWPOINT_2M,
        "relative_humidity_2m": WeatherField.RELATIVE_HUMIDITY_2M,
        "pressure_msl": WeatherField.PRESSURE_MSL,
        "wind_speed_10m": WeatherField.WIND_SPEED_10M,
    }
)

DEFAULT_WEATHER_PALETTES: Mapping[WeatherField, str] = MappingProxyType(
    {palette.field: palette.id for palette in WEATHER_PALETTES.values()}
)


def palettes_for_field(field: WeatherField) -> tuple[WeatherPalette, ...]:
    normalized = WeatherField(field)
    return tuple(
        palette
        for palette in WEATHER_PALETTES.values()
        if palette.field is normalized
    )
