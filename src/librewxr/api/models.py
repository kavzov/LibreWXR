# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
from typing import Any, Literal

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


class RadarAnimationData(BaseModel):
    substeps: int
    past: list[RadarTimestamp]
    nowcast: list[RadarTimestamp]


class RadarData(BaseModel):
    past: list[RadarTimestamp]
    nowcast: list[RadarTimestamp]
    animation: RadarAnimationData | None = None
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
    point_url_template: str
    sizes: list[int]
    formats: list[str]
    min_zoom: int
    max_zoom: int


class WeatherPointResponse(BaseModel):
    field: str
    timestamp: int
    latitude: float
    longitude: float
    value: float | None
    unit: str
    active_model_run: str | None
    stale: bool


class RadarPointNowcastFrame(BaseModel):
    time: int
    minutes_offset: int
    period: Literal["observed", "forecast"]
    coverage: Literal["in_range", "out_of_range"]
    region: str | None
    sample_count: int
    wet_pixel_count: int
    wet_fraction: float | None
    max_dbz: float | None
    max_rate_mmh: float | None
    blend_weight: float


class RadarPointNowcastResponse(BaseModel):
    generated: int
    latitude: float
    longitude: float
    radius_km: float
    noise_floor_dbz: float
    latest_observation_time: int
    latest_age_seconds: int
    stale: bool
    history_minutes_available: int
    forecast_minutes_available: int
    frames: list[RadarPointNowcastFrame]
