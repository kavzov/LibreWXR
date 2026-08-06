# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
from typing import Any

from pydantic import BaseModel


class AlertProperties(BaseModel):
    title: str
    severity: str
    time: int | None
    expires: int | None
    description: str
    regions: list[str]
    uri: str


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    properties: AlertProperties
    geometry: dict[str, Any] | None


class AlertsResponse(BaseModel):
    type: str = "FeatureCollection"
    features: list[GeoJSONFeature]


class RadarTimestamp(BaseModel):
    time: int
    path: str


class ColorScheme(BaseModel):
    id: int
    name: str


class RadarData(BaseModel):
    past: list[RadarTimestamp]
    nowcast: list[RadarTimestamp]
    colorSchemes: list[ColorScheme]


class SatelliteData(BaseModel):
    infrared: list[RadarTimestamp]


class WeatherMapsResponse(BaseModel):
    version: str
    generated: int
    host: str
    radar: RadarData
    satellite: SatelliteData


class WeatherPaletteStop(BaseModel):
    value: float
    color: str


class WeatherPaletteInfo(BaseModel):
    id: str
    display_name: str
    unit: str
    minimum: float
    maximum: float
    below_color: str
    above_color: str
    nodata_color: str
    opacity: float
    stops: list[WeatherPaletteStop]


class WeatherFieldInfo(BaseModel):
    id: str
    display_name: str
    unit: str
    palette_ids: list[str]


class WeatherMetadataResponse(BaseModel):
    active_model_run: str | None
    generated: int
    stale: bool
    attribution: str
    fields: list[WeatherFieldInfo]
    available_timestamps: list[int]
    default_timestamp: int | None
    palette_ids: list[str]
    palettes: list[WeatherPaletteInfo]
    tile_url_template: str
    sizes: list[int]
    formats: list[str]
    min_zoom: int
    max_zoom: int
