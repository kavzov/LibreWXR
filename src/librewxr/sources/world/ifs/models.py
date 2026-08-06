# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Immutable frame records used by the global ECMWF IFS grid."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from librewxr.data.weather_fields import WeatherField


ECMWF_STATE_FORMAT_VERSION = 2


@dataclass(frozen=True)
class WeatherFrame:
    """One native or precipitation-interpolated ECMWF valid time.

    Every value in ``fields`` uses the compact encoded dtype declared by its
    :class:`~librewxr.data.weather_fields.FieldSpec`. Native model frames carry
    all configured scalar/vector fields. Synthetic precipitation frames carry
    precipitation only so optical-flow interpolation never materialises full
    global grids for unrelated continuous fields.
    """

    timestamp: int
    fields: Mapping[WeatherField, np.ndarray]
    snow_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        normalized = {
            WeatherField(field): values for field, values in self.fields.items()
        }
        object.__setattr__(self, "fields", MappingProxyType(normalized))

    def has_field(self, field: WeatherField) -> bool:
        return WeatherField(field) in self.fields

    def field(self, field: WeatherField) -> np.ndarray:
        return self.fields[WeatherField(field)]
