# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Global ECMWF IFS precipitation and physical weather-field storage."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import fsspec
import numpy as np
from earthkit.regrid import interpolate
from omfiles import OmFileReader

from librewxr.config import settings
from librewxr.data.weather_sampling import (
    SamplingPlan,
    build_regular_sampling_plan,
    cached_regular_tile_sampling_plan,
    web_mercator_tile_latlons,
)
from librewxr.data.weather_fields import (
    WeatherField,
    decode_field,
    encode_field,
    field_spec,
    relative_humidity_from_temperature_dewpoint,
    wind_speed_from_uv,
)
from librewxr.sources.world.ifs.models import (
    ECMWF_STATE_FORMAT_VERSION,
    WeatherFrame,
)

logger = logging.getLogger(__name__)

# Regridded output at 0.1° resolution
PIXEL_SIZE = 0.1
WEST = -180.0
EAST = 180.0
NORTH = 90.0
SOUTH = -90.0
GRID_WIDTH = int((EAST - WEST) / PIXEL_SIZE)  # 3600
GRID_HEIGHT = int((NORTH - SOUTH) / PIXEL_SIZE) + 1  # 1801
GRID_SHAPE = (GRID_HEIGHT, GRID_WIDTH)
GRID_GEOMETRY_VERSION = 1

# Z-R relationship constants (Marshall-Palmer)
ZR_A_RAIN = 200.0
ZR_B_RAIN = 1.6
ZR_A_SNOW = 2000.0
ZR_B_SNOW = 2.0

S3_LATEST_PATH = "data_spatial/ecmwf_ifs/latest.json"
MEMMAP_FORMAT_VERSION = 2
ACTIVE_MANIFEST_VERSION = 1
ACTIVE_MANIFEST_NAME = "active.json"
RUN_DIRECTORY_NAME = "runs"

# Exact Open-Meteo data_spatial child names verified against the IFS .om files.
IFS_FIELD_VARIABLES: dict[WeatherField, str] = {
    WeatherField.TEMPERATURE_2M: "temperature_2m",
    WeatherField.DEWPOINT_2M: "dew_point_2m",
    WeatherField.PRESSURE_MSL: "pressure_msl",
    WeatherField.WIND_U_10M: "wind_u_component_10m",
    WeatherField.WIND_V_10M: "wind_v_component_10m",
}
REQUIRED_WEATHER_FIELDS = frozenset(IFS_FIELD_VARIABLES)
REQUIRED_SOURCE_VARIABLES = frozenset(IFS_FIELD_VARIABLES.values())


class _WeatherFrameDict(dict[int, WeatherFrame]):
    """Named-frame mapping with a narrow legacy test/integration adapter.

    Assigning an old ``(precip, snow)`` pair is converted immediately; tuples
    are never retained internally. This softens the private representation
    migration for downstream integrations that used test-style injection.
    """

    def __setitem__(self, timestamp: int, frame: WeatherFrame) -> None:
        if isinstance(frame, tuple) and len(frame) == 2:
            precipitation, snow = frame
            frame = WeatherFrame(
                timestamp,
                {WeatherField.PRECIPITATION: precipitation},
                snow,
            )
        if not isinstance(frame, WeatherFrame):
            raise TypeError("ECMWF timesteps must contain WeatherFrame values")
        super().__setitem__(timestamp, frame)


class ECMWFGrid:
    """ECMWF IFS 9 km global fallback and generic weather-field source.

    Precipitation keeps the legacy nearest-time and optical-flow behaviour used
    by nowcast and radar rendering. Instantaneous physical fields are stored at
    their native valid times and linearly interpolated only after the requested
    coordinates have been sampled.
    """

    name = "ecmwf_ifs"
    global_catch_all = True

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        cleanup_tmp: bool = True,
    ):
        self._timesteps: dict[int, WeatherFrame] = _WeatherFrameDict()
        self._grid_version = GRID_GEOMETRY_VERSION
        self._content_version = 0
        self._reference_time: str | None = None
        self._previous_reference_time: str | None = None
        self._last_modified_time: str | None = None
        self._last_successful_update: int | None = None
        self._last_update_error: str | None = None
        self._last_check_time: int | None = None
        self._weather_fields = frozenset(settings.get_weather_fields())
        self._fs: fsspec.AbstractFileSystem | None = None
        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "ecmwf_ifs"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_ecmwf_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        self._runs_dir = self._memmap_dir / RUN_DIRECTORY_NAME
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._memmap_dir / ACTIVE_MANIFEST_NAME
        if cleanup_tmp:
            for path in self._memmap_dir.rglob("*.tmp"):
                path.unlink(missing_ok=True)
        if self._persistent:
            self._restore_last_known_good()
            if cleanup_tmp:
                self._cleanup_run_directories(retain_newest_unprotected=1)
        logger.info(
            "ECMWF memmap directory: %s (persistent=%s, restored=%s)",
            self._memmap_dir,
            self._persistent,
            self._reference_time or "none",
        )

    @staticmethod
    def _run_key(reference_time: str) -> str:
        return (
            reference_time.replace("+00:00", "Z")
            .replace("-", "")
            .replace(":", "")
        )

    def _array_basename(
        self,
        reference_time: str,
        timestamp: int,
        name: str,
    ) -> str:
        return (
            f"v{MEMMAP_FORMAT_VERSION}_r{self._run_key(reference_time)}"
            f"_t{timestamp}_{name}.dat"
        )

    def _array_path(
        self,
        reference_time: str,
        timestamp: int,
        name: str,
    ) -> Path:
        return self._run_dir(reference_time) / self._array_basename(
            reference_time, timestamp, name
        )

    def _run_dir(self, reference_time: str) -> Path:
        path = self._runs_dir / self._run_key(reference_time)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _expected_size(dtype: np.dtype, shape: tuple[int, ...]) -> int:
        return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize

    def _open_memmap(
        self,
        path: Path,
        dtype: np.dtype,
        shape: tuple[int, ...] | None = None,
    ) -> np.memmap | None:
        if shape is None:
            shape = GRID_SHAPE
        try:
            if path.stat().st_size != self._expected_size(dtype, shape):
                logger.warning("ECMWF cached array has wrong size: %s", path)
                return None
        except FileNotFoundError:
            return None
        return np.memmap(path, dtype=dtype, mode="r", shape=shape)

    def _to_memmap_path(self, final: Path, data: np.ndarray) -> np.memmap:
        """Atomically persist ``data`` at an explicit final path."""

        final.parent.mkdir(parents=True, exist_ok=True)
        expected = self._expected_size(data.dtype, data.shape)
        existing = self._open_memmap(final, data.dtype, data.shape)
        if existing is not None:
            return existing

        tmp = final.with_suffix(final.suffix + ".tmp")
        tmp.unlink(missing_ok=True)
        mm = np.memmap(tmp, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        actual = tmp.stat().st_size
        if actual != expected:
            tmp.unlink(missing_ok=True)
            raise IOError(
                f"ECMWF memmap size mismatch for {tmp}: {actual} != {expected}"
            )
        os.replace(tmp, final)
        reopened = self._open_memmap(final, data.dtype, data.shape)
        if reopened is None:
            raise IOError(f"Failed to verify ECMWF memmap {final}")
        return reopened

    def _to_memmap(self, name: str, data: np.ndarray) -> np.memmap:
        """Atomically persist one array and reopen it read-only.

        ``name`` may be a legacy stem used by existing persistence tests or a
        complete versioned stem generated by :meth:`_array_basename`.
        """

        return self._to_memmap_path(self._memmap_dir / f"{name}.dat", data)

    def _persist_frame(
        self,
        reference_time: str,
        frame: WeatherFrame,
    ) -> WeatherFrame:
        fields: dict[WeatherField, np.ndarray] = {}
        for field, values in frame.fields.items():
            fields[field] = self._to_memmap_path(
                self._array_path(reference_time, frame.timestamp, field.value),
                values,
            )
        snow = None
        if frame.snow_mask is not None:
            snow = self._to_memmap_path(
                self._array_path(reference_time, frame.timestamp, "snow_mask"),
                frame.snow_mask,
            )
        return WeatherFrame(frame.timestamp, fields, snow)

    def _open_cached_frame(
        self,
        reference_time: str,
        timestamp: int,
        required_fields: frozenset[WeatherField],
        require_snow: bool,
    ) -> WeatherFrame | None:
        fields: dict[WeatherField, np.ndarray] = {}
        for field in required_fields:
            spec = field_spec(field)
            path = self._array_path(reference_time, timestamp, field.value)
            values = self._open_memmap(path, spec.storage_dtype)
            if values is None:
                return None
            fields[field] = values
        snow = None
        if require_snow:
            snow = self._open_memmap(
                self._array_path(reference_time, timestamp, "snow_mask"),
                np.dtype(bool),
            )
            if snow is None:
                return None
        return WeatherFrame(timestamp, fields, snow)

    def _frame_files(self, frames: Iterable[WeatherFrame]) -> set[str]:
        names: set[str] = set()
        for frame in frames:
            arrays = list(frame.fields.values())
            if frame.snow_mask is not None:
                arrays.append(frame.snow_mask)
            for values in arrays:
                filename = getattr(values, "filename", None)
                if filename is not None:
                    path = Path(str(filename))
                    try:
                        names.add(path.relative_to(self._memmap_dir).as_posix())
                    except ValueError:
                        names.add(path.name)
        return names

    @staticmethod
    def _stored_array_valid(values: np.ndarray) -> bool:
        filename = getattr(values, "filename", None)
        if filename is None:
            return True
        try:
            return Path(filename).stat().st_size == values.nbytes
        except FileNotFoundError:
            return False

    @classmethod
    def _frame_storage_valid(
        cls,
        frame: WeatherFrame,
        required_fields: frozenset[WeatherField],
        require_snow: bool,
    ) -> bool:
        if not required_fields <= frame.fields.keys():
            return False
        if any(
            not cls._stored_array_valid(frame.field(field))
            for field in required_fields
        ):
            return False
        return not require_snow or (
            frame.snow_mask is not None
            and cls._stored_array_valid(frame.snow_mask)
        )

    def _cleanup_memmap_files(self, keep: set[str] | None = None) -> None:
        """Remove inactive ``.dat`` files, never arrays in ``keep``."""

        for path in self._memmap_dir.rglob("*.dat"):
            try:
                relative = path.relative_to(self._memmap_dir).as_posix()
            except ValueError:
                relative = path.name
            if keep is not None and (relative in keep or path.name in keep):
                continue
            try:
                path.unlink()
            except OSError:
                logger.debug("Could not remove stale ECMWF memmap %s", path)

    def _cleanup_run_directories(
        self,
        *,
        retain_newest_unprotected: int = 0,
        extra_protected: Iterable[str] = (),
    ) -> None:
        """Retain active/previous generations and remove older run trees."""

        protected = {
            self._run_key(reference)
            for reference in (
                self._reference_time,
                self._previous_reference_time,
                *extra_protected,
            )
            if reference
        }
        candidates = [path for path in self._runs_dir.iterdir() if path.is_dir()]
        unprotected = sorted(
            (path for path in candidates if path.name not in protected),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        retained = set(unprotected[:retain_newest_unprotected])
        for path in unprotected:
            if path in retained:
                continue
            try:
                shutil.rmtree(path)
            except OSError:
                logger.debug("Could not remove stale ECMWF run %s", path)

    def _state_for_frames(
        self,
        frames: dict[int, WeatherFrame],
        *,
        reference_time: str | None,
        previous_reference_time: str | None,
        last_modified_time: str | None,
        last_successful_update: int | None,
        last_update_error: str | None,
        content_version: int,
    ) -> dict:
        timesteps_state: dict[str, dict] = {}
        for ts, frame in frames.items():
            timesteps_state[str(ts)] = {
                "timestamp": frame.timestamp,
                "fields": {
                    field.value: self._descriptor(values)
                    for field, values in frame.fields.items()
                },
                "snow_mask": (
                    self._descriptor(frame.snow_mask)
                    if frame.snow_mask is not None
                    else None
                ),
            }
        return {
            "format_version": ECMWF_STATE_FORMAT_VERSION,
            "grid_version": self._grid_version,
            "content_version": content_version,
            "memmap_dir": str(self._memmap_dir),
            "reference_time": reference_time,
            "previous_reference_time": previous_reference_time,
            "last_modified_time": last_modified_time,
            "last_successful_update": last_successful_update,
            "last_check_time": self._last_check_time,
            "last_update_error": last_update_error,
            "timesteps": timesteps_state,
        }

    def _write_active_manifest(
        self,
        frames: dict[int, WeatherFrame],
        *,
        reference_time: str,
        previous_reference_time: str | None,
        last_modified_time: str | None,
        published_at: int,
        content_version: int,
    ) -> None:
        """Fsync and atomically replace the last-known-good run manifest."""

        if not self._persistent:
            return
        payload = {
            "version": ACTIVE_MANIFEST_VERSION,
            "active_run": reference_time,
            "previous_run": previous_reference_time,
            "published_at": published_at,
            "store_state": self._state_for_frames(
                frames,
                reference_time=reference_time,
                previous_reference_time=previous_reference_time,
                last_modified_time=last_modified_time,
                last_successful_update=published_at,
                last_update_error=None,
                content_version=content_version,
            ),
        }
        tmp = self._manifest_path.with_name(f".{ACTIVE_MANIFEST_NAME}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._manifest_path)

    def _restore_last_known_good(self) -> bool:
        """Restore the active manifest, falling back to legacy state.json."""

        state = None
        source = None
        if self._manifest_path.exists():
            try:
                payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                if payload.get("version") != ACTIVE_MANIFEST_VERSION:
                    raise ValueError("unsupported active manifest version")
                state = payload["store_state"]
                source = self._manifest_path
            except Exception:
                logger.exception("Could not read ECMWF active manifest")

        if state is None:
            legacy_state = self._memmap_dir.parent / "state.json"
            if legacy_state.exists():
                try:
                    envelope = json.loads(legacy_state.read_text(encoding="utf-8"))
                    state = envelope.get("stores", {}).get("ecmwf_grid")
                    source = legacy_state if state is not None else None
                except Exception:
                    logger.exception("Could not read ECMWF state.json fallback")

        if state is None:
            return False
        try:
            self.__setstate__(state)
            logger.info(
                "Restored ECMWF last-known-good run %s from %s",
                self._reference_time,
                source,
            )
            return True
        except Exception:
            logger.exception("Could not restore ECMWF last-known-good run")
            self._timesteps = _WeatherFrameDict()
            self._reference_time = None
            self._previous_reference_time = None
            return False

    def _publish_prepared(
        self,
        frames: dict[int, WeatherFrame],
        *,
        reference_time: str,
        last_modified_time: str | None,
    ) -> bool:
        """Publish a complete prepared generation in memory and on disk."""

        new_active = _WeatherFrameDict()
        for ts, frame in sorted(frames.items()):
            new_active[ts] = frame
        previous = self._previous_reference_time
        if self._reference_time and self._reference_time != reference_time:
            previous = self._reference_time
        published_at = int(time.time())
        content_version = self._content_version + 1
        try:
            self._write_active_manifest(
                new_active,
                reference_time=reference_time,
                previous_reference_time=previous,
                last_modified_time=last_modified_time,
                published_at=published_at,
                content_version=content_version,
            )
        except Exception as exc:
            return self._fail_update(f"active manifest publication failed: {exc}")

        # One mapping assignment publishes every field together. Readers that
        # already captured the old mapping keep valid memmap references.
        self._timesteps = new_active
        self._reference_time = reference_time
        self._previous_reference_time = previous
        self._last_modified_time = last_modified_time
        self._last_successful_update = published_at
        self._last_update_error = None
        self._content_version = content_version
        self._cleanup_run_directories()
        return True

    def _precip_timestamps(
        self, frames: dict[int, WeatherFrame] | None = None
    ) -> list[int]:
        frames = self._timesteps if frames is None else frames
        return sorted(
            ts
            for ts, frame in frames.items()
            if frame.has_field(WeatherField.PRECIPITATION)
        )

    def _field_timestamps(
        self,
        field: WeatherField,
        frames: dict[int, WeatherFrame] | None = None,
    ) -> list[int]:
        frames = self._timesteps if frames is None else frames
        return sorted(
            ts for ts, frame in frames.items() if frame.has_field(field)
        )

    @property
    def _sorted_timestamps(self) -> list[int]:
        """Legacy private view: precipitation-valid timestamps only."""

        return self._precip_timestamps()

    @_sorted_timestamps.setter
    def _sorted_timestamps(self, _value: list[int]) -> None:
        # Older tests and integrations assigned this cache explicitly. The list
        # is now derived from named frames, so the assignment is unnecessary.
        return None

    @property
    def data(self) -> np.ndarray | None:
        frames = self._timesteps
        timestamps = self._precip_timestamps(frames)
        if not timestamps:
            return None
        return frames[timestamps[-1]].field(WeatherField.PRECIPITATION)

    @property
    def reference_time(self) -> str | None:
        return self._reference_time

    @property
    def previous_model_run(self) -> str | None:
        return self._previous_reference_time

    @property
    def timestep_count(self) -> int:
        """Legacy precipitation frame count (including optical-flow frames)."""

        return len(self._precip_timestamps())

    def available_timestamps(self) -> list[int]:
        """Native valid times shared by the atomically published weather fields."""

        for field in sorted(self._weather_fields, key=lambda item: item.value):
            timestamps = self._field_timestamps(field)
            if timestamps:
                return timestamps
        return []

    def default_timestamp(self, now: int | None = None) -> int | None:
        """Return the available valid time nearest to ``now``."""

        timestamps = self.available_timestamps()
        if not timestamps:
            return None
        target = int(time.time()) if now is None else now
        return min(timestamps, key=lambda value: abs(value - target))

    @property
    def model_version(self) -> str:
        """Version token for rendered-tile cache isolation."""

        return (
            f"{self._reference_time or 'unpublished'}:"
            f"g{self._grid_version}:c{self._content_version}"
        )

    @property
    def data_bytes(self) -> int:
        return sum(
            values.nbytes
            for frame in self._timesteps.values()
            for values in (
                *frame.fields.values(),
                *(() if frame.snow_mask is None else (frame.snow_mask,)),
            )
        )

    @property
    def field_bytes(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for frame in self._timesteps.values():
            for field, values in frame.fields.items():
                totals[field.value] = totals.get(field.value, 0) + values.nbytes
            if frame.snow_mask is not None:
                totals["snow_mask"] = totals.get("snow_mask", 0) + frame.snow_mask.nbytes
        return totals

    def available_fields(self) -> frozenset[WeatherField]:
        fields: set[WeatherField] = {
            WeatherField.PRECIPITATION,
            *self._weather_fields,
        }
        for candidate in WeatherField:
            spec = field_spec(candidate)
            if spec.derived and set(spec.dependencies) <= fields:
                fields.add(candidate)
        return frozenset(fields)

    def has_field(self, field: WeatherField) -> bool:
        try:
            return WeatherField(field) in self.available_fields()
        except ValueError:
            return False

    def _get_fs(self) -> fsspec.AbstractFileSystem:
        if self._fs is None:
            self._fs = fsspec.filesystem(
                "s3",
                anon=True,
                client_kwargs={"region_name": settings.ecmwf_s3_region},
            )
        return self._fs

    @staticmethod
    def _nearest_from(
        timestamps: list[int], timestamp: int | None
    ) -> int | None:
        if not timestamps:
            return None
        if timestamp is None:
            return timestamps[-1]
        idx = int(np.searchsorted(timestamps, timestamp))
        if idx == 0:
            return timestamps[0]
        if idx >= len(timestamps):
            return timestamps[-1]
        before = timestamps[idx - 1]
        after = timestamps[idx]
        return before if timestamp - before <= after - timestamp else after

    def _nearest_timestamp(self, timestamp: int | None) -> int | None:
        return self._nearest_from(self._precip_timestamps(), timestamp)

    @staticmethod
    def _vt_to_unix(vt: str) -> int:
        vt_dt = datetime.fromisoformat(vt.replace("Z", "+00:00"))
        if vt_dt.tzinfo is None:
            vt_dt = vt_dt.replace(tzinfo=timezone.utc)
        return int(vt_dt.timestamp())

    @staticmethod
    def _select_valid_times(valid_times: list[str], max_ts: int) -> list[str]:
        """Preserve the existing radar/nowcast precipitation window."""

        if len(valid_times) <= max_ts:
            return valid_times
        now_ts = int(datetime.now(timezone.utc).timestamp())
        anchor_target = now_ts
        if settings.nowcast_enabled:
            anchor_target += settings.nowcast_frames * settings.fetch_interval
        vt_unix = [ECMWFGrid._vt_to_unix(vt) for vt in valid_times]
        anchor_idx = next(
            (i for i, value in enumerate(vt_unix) if value >= anchor_target),
            len(valid_times) - 1,
        )
        end = anchor_idx + 1
        start = max(end - max_ts, 0)
        end = min(start + max_ts, len(valid_times))
        return valid_times[start:end]

    @staticmethod
    def _select_weather_valid_times(
        valid_times: list[str],
        forecast_hours: int,
        max_timesteps: int,
        now_ts: int | None = None,
    ) -> list[str]:
        """Select native valid times from nearest past through the horizon."""

        if not valid_times:
            return []
        if now_ts is None:
            now_ts = int(datetime.now(timezone.utc).timestamp())
        pairs = sorted((ECMWFGrid._vt_to_unix(vt), vt) for vt in valid_times)
        past = [pair for pair in pairs if pair[0] <= now_ts]
        selected: list[tuple[int, str]] = [past[-1]] if past else [pairs[0]]
        horizon = now_ts + forecast_hours * 3600
        selected.extend(
            pair
            for pair in pairs
            if selected[0][0] < pair[0] <= horizon
        )
        # Stable de-duplication for the all-future case.
        selected = list(dict.fromkeys(selected))
        if max_timesteps > 0:
            selected = selected[:max_timesteps]
        return [vt for _ts, vt in selected]

    def _fail_update(self, message: str) -> bool:
        self._last_update_error = message
        now = int(time.time())
        stale_age = (
            now - self._vt_to_unix(self._reference_time)
            if self._reference_time
            else None
        )
        suffix = (
            f" (active run age {max(stale_age, 0) / 3600:.1f}h)"
            if stale_age is not None
            else ""
        )
        logger.warning(
            "ECMWF IFS: %s; keeping previous complete run%s", message, suffix
        )
        if self._persistent:
            self._cleanup_run_directories(retain_newest_unprotected=1)
        return False

    async def fetch(self) -> bool:
        try:
            return await asyncio.to_thread(self._fetch_sync)
        except Exception as exc:
            logger.exception("Unexpected error fetching ECMWF IFS data")
            return self._fail_update(f"unexpected update error: {exc}")

    def _fetch_sync(self) -> bool:
        from librewxr.data.retry import retry_sync

        self._last_check_time = int(time.time())
        fs = self._get_fs()
        bucket = settings.ecmwf_s3_bucket
        latest_raw = retry_sync(
            fs.cat,
            f"{bucket}/{S3_LATEST_PATH}",
            log_name="ECMWF IFS latest.json",
        )
        if latest_raw is None:
            return self._fail_update("failed to fetch latest.json after retries")
        try:
            latest = json.loads(latest_raw)
        except (TypeError, json.JSONDecodeError) as exc:
            return self._fail_update(f"invalid latest.json: {exc}")
        if not latest.get("completed", False):
            return self._fail_update("model run is not complete")

        ref_time = latest.get("reference_time")
        valid_times = latest.get("valid_times", [])
        variables = set(latest.get("variables", []))
        if not isinstance(ref_time, str):
            return self._fail_update("latest.json has no reference_time")
        required_source_variables = {
            IFS_FIELD_VARIABLES[field] for field in self._weather_fields
        }
        missing_variables = sorted(required_source_variables - variables)
        if missing_variables:
            return self._fail_update(
                "missing required variables: " + ", ".join(missing_variables)
            )
        if "precipitation" not in variables:
            return self._fail_update("missing required variable: precipitation")
        if len(valid_times) < 2:
            return self._fail_update("fewer than two valid times published")

        if self._reference_time:
            try:
                if self._vt_to_unix(ref_time) < self._vt_to_unix(self._reference_time):
                    logger.warning(
                        "ECMWF IFS metadata regressed from %s to %s; "
                        "keeping last-known-good run",
                        self._reference_time,
                        ref_time,
                    )
                    return True
            except ValueError:
                pass

        precip_vts = self._select_valid_times(
            valid_times[1:], settings.get_ecmwf_max_timesteps()
        )
        weather_vts = (
            self._select_weather_valid_times(
                valid_times,
                settings.weather_fields_forecast_hours,
                settings.weather_fields_max_timesteps,
            )
            if self._weather_fields
            else []
        )
        precip_set = set(precip_vts)
        weather_set = set(weather_vts)
        selected_vts = sorted(
            precip_set | weather_set,
            key=self._vt_to_unix,
        )
        if not selected_vts:
            return self._fail_update("no valid times selected")

        desired_complete = True
        if ref_time != self._reference_time:
            desired_complete = False
        else:
            for vt in selected_vts:
                ts = self._vt_to_unix(vt)
                frame = self._timesteps.get(ts)
                required = self._weather_fields if vt in weather_set else frozenset()
                if vt in precip_set:
                    required = frozenset({*required, WeatherField.PRECIPITATION})
                if frame is None or not self._frame_storage_valid(
                    frame, required, require_snow=vt in precip_set
                ):
                    desired_complete = False
                    break
        if desired_complete:
            self._last_update_error = None
            logger.debug("ECMWF IFS: run %s and requested windows unchanged", ref_time)
            return True

        ref_dt = datetime.fromisoformat(ref_time.replace("Z", "+00:00"))
        run_prefix = (
            f"{bucket}/{settings.ecmwf_s3_prefix}"
            f"/{ref_dt.year}/{ref_dt.month:02d}/{ref_dt.day:02d}"
            f"/{ref_dt.hour:02d}{ref_dt.minute:02d}Z"
        )
        has_snow = "snowfall_water_equivalent" in variables
        logger.info(
            "Preparing ECMWF IFS run %s: %d weather valid times, "
            "%d precipitation valid times",
            ref_time,
            len(weather_vts),
            len(precip_vts),
        )

        prepared: dict[int, WeatherFrame] = {}
        missing: list[tuple[str, bool, frozenset[WeatherField]]] = []
        for vt in selected_vts:
            ts = self._vt_to_unix(vt)
            required = self._weather_fields if vt in weather_set else frozenset()
            need_precip = vt in precip_set
            if need_precip:
                required = frozenset({*required, WeatherField.PRECIPITATION})

            active = self._timesteps.get(ts) if ref_time == self._reference_time else None
            if active is not None and self._frame_storage_valid(
                active, required, require_snow=need_precip
            ):
                prepared[ts] = active
                continue
            cached = self._open_cached_frame(
                ref_time, ts, required, require_snow=need_precip
            )
            if cached is not None:
                prepared[ts] = cached
            else:
                missing.append((vt, need_precip, required))

        failures: list[str] = []
        if missing:
            worker_count = max(
                1, min(len(missing), settings.nwp_fetch_concurrency)
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        self._fetch_and_persist_timestep,
                        fs,
                        run_prefix,
                        ref_time,
                        vt,
                        need_precip,
                        has_snow,
                        required,
                    ): vt
                    for vt, need_precip, required in missing
                }
                for future in as_completed(futures):
                    vt = futures[future]
                    try:
                        persisted = future.result()
                        prepared[persisted.timestamp] = persisted
                    except Exception as exc:
                        failures.append(f"{vt}: {exc}")
                        logger.warning(
                            "ECMWF required timestep %s failed", vt, exc_info=True
                        )
        if failures:
            return self._fail_update(
                "required timestep preparation failed (" + "; ".join(failures) + ")"
            )

        for vt in weather_vts:
            frame = prepared.get(self._vt_to_unix(vt))
            if frame is None or not self._weather_fields <= frame.fields.keys():
                return self._fail_update(f"incomplete weather frame at {vt}")
        for vt in precip_vts:
            frame = prepared.get(self._vt_to_unix(vt))
            if frame is None or not frame.has_field(WeatherField.PRECIPITATION):
                return self._fail_update(f"incomplete precipitation frame at {vt}")

        if settings.ecmwf_interpolation:
            try:
                from librewxr.sources.world.ifs.interpolation import (
                    interpolate_timesteps,
                )

                prepared = interpolate_timesteps(prepared)
                prepared = {
                    ts: self._persist_frame(ref_time, frame)
                    for ts, frame in prepared.items()
                }
            except Exception as exc:
                logger.warning("ECMWF precipitation interpolation failed", exc_info=True)
                return self._fail_update(f"precipitation interpolation failed: {exc}")

        if not self._publish_prepared(
            prepared,
            reference_time=ref_time,
            last_modified_time=latest.get("last_modified_time"),
        ):
            return False
        logger.info(
            "ECMWF IFS atomically published: ref=%s, weather=%d, precip=%d",
            ref_time,
            len(self._field_timestamps(WeatherField.TEMPERATURE_2M)),
            self.timestep_count,
        )
        return True

    def _fetch_and_persist_timestep(
        self,
        fs: fsspec.AbstractFileSystem,
        run_prefix: str,
        reference_time: str,
        vt: str,
        need_precip: bool,
        has_snow: bool,
        required_fields: frozenset[WeatherField],
    ) -> WeatherFrame:
        """Decode and persist inside one worker to bound heap residency."""

        frame = self._fetch_one_timestep(
            fs,
            run_prefix,
            vt,
            need_precip,
            has_snow,
            required_fields,
        )
        self._validate_frame(frame)
        return self._persist_frame(reference_time, frame)

    @staticmethod
    def _read_regridded(reader: OmFileReader, variable_name: str) -> np.ndarray:
        child = reader.get_child_by_name(variable_name)
        try:
            raw = child[:].reshape(-1).astype(np.float32)
        finally:
            child.close()
        if not np.isfinite(raw).any():
            raise ValueError(f"{variable_name} contains no finite values")
        grid = interpolate(
            raw,
            in_grid={"grid": "O1280"},
            out_grid={"grid": [PIXEL_SIZE, PIXEL_SIZE]},
            method="linear",
        )
        grid = np.asarray(grid, dtype=np.float32)
        if grid.shape != GRID_SHAPE:
            raise ValueError(
                f"{variable_name} regrid shape {grid.shape} != {GRID_SHAPE}"
            )
        return np.roll(grid, GRID_WIDTH // 2, axis=1)

    def _fetch_one_timestep(
        self,
        fs: fsspec.AbstractFileSystem,
        run_prefix: str,
        vt: str,
        need_precip: bool,
        has_snow: bool,
        required_fields: frozenset[WeatherField],
    ) -> WeatherFrame:
        """Read one .om object, regrid, unit-convert, and compactly encode it."""

        vt_clean = vt.replace("Z", "").replace(":", "")
        om_path = f"{run_prefix}/{vt_clean}.om"
        from librewxr.data.retry import retry_sync

        reader = retry_sync(
            OmFileReader.from_fsspec,
            fs,
            om_path,
            log_name=f"ECMWF IFS {vt}",
        )
        if reader is None:
            raise RuntimeError(f"failed to open {om_path} after retries")

        fields: dict[WeatherField, np.ndarray] = {}
        snow_mask = None
        try:
            for field in REQUIRED_WEATHER_FIELDS:
                if field not in required_fields:
                    continue
                physical = self._read_regridded(reader, IFS_FIELD_VARIABLES[field])
                if field is WeatherField.PRESSURE_MSL:
                    physical = physical / 100.0  # Open-Meteo .om stores Pa.
                fields[field] = encode_field(field, physical)

            if need_precip:
                precip_grid = self._read_regridded(reader, "precipitation")
                if has_snow:
                    snow_grid = self._read_regridded(
                        reader, "snowfall_water_equivalent"
                    )
                else:
                    snow_grid = np.zeros_like(precip_grid)
                precipitation, snow_mask = self._encode_precipitation(
                    precip_grid, snow_grid, vt
                )
                fields[WeatherField.PRECIPITATION] = precipitation
        finally:
            reader.close()

        return WeatherFrame(self._vt_to_unix(vt), fields, snow_mask)

    @staticmethod
    def _encode_precipitation(
        precip_grid: np.ndarray,
        snow_grid: np.ndarray,
        vt: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        rate = np.maximum(precip_grid, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            snow_ratio = np.where(
                rate > 1e-6,
                np.clip(snow_grid / rate, 0.0, 1.0),
                0.0,
            )
        is_snow = snow_ratio > settings.ecmwf_snow_ratio_threshold
        z_values = np.where(
            is_snow,
            ZR_A_SNOW * np.power(np.maximum(rate, 1e-10), ZR_B_SNOW),
            ZR_A_RAIN * np.power(np.maximum(rate, 1e-10), ZR_B_RAIN),
        )
        dbz = np.where(
            rate > 0.01,
            10.0 * np.log10(np.maximum(z_values, 1e-10)),
            0.0,
        )
        result = np.clip((dbz + 32.0) * 2.0, 0, 255).astype(np.uint8)
        result[rate <= 0.01] = 0
        valid_pixels = rate > 0.01
        logger.debug(
            "Timestep %s: %.1f-%.1f dBZ, %d precip pixels, %.1f%% snow",
            vt,
            dbz[valid_pixels].min() if valid_pixels.any() else 0,
            dbz[valid_pixels].max() if valid_pixels.any() else 0,
            int(valid_pixels.sum()),
            100.0 * (is_snow & valid_pixels).sum() / max(1, valid_pixels.sum()),
        )
        return result, is_snow

    @staticmethod
    def _validate_frame(frame: WeatherFrame) -> None:
        for field, values in frame.fields.items():
            spec = field_spec(field)
            if values.shape != GRID_SHAPE:
                raise ValueError(f"{field.value} has invalid shape {values.shape}")
            if values.dtype != spec.storage_dtype:
                raise ValueError(f"{field.value} has invalid dtype {values.dtype}")
            if spec.nodata is not None and np.all(values == spec.nodata):
                raise ValueError(f"{field.value} contains only nodata")
        if frame.snow_mask is not None:
            if frame.snow_mask.shape != GRID_SHAPE or frame.snow_mask.dtype != bool:
                raise ValueError("snow_mask has invalid shape or dtype")

    @property
    def grid_version(self) -> int:
        """Version of the spatial geometry used in sampling-plan cache keys."""

        return self._grid_version

    @property
    def sampling_grid_identity(self) -> str:
        """Stable identity for the regular IFS output grid."""

        return f"{self.name}:regular_global"

    def invalidate_sampling_plans(self) -> None:
        """Advance the geometry version so old cached plans cannot be reused."""

        self._grid_version += 1

    @staticmethod
    def _build_sampling_plan(lat: np.ndarray, lon: np.ndarray) -> SamplingPlan:
        return build_regular_sampling_plan(
            lat,
            lon,
            west=WEST,
            north=NORTH,
            pixel_size_x=PIXEL_SIZE,
            pixel_size_y=PIXEL_SIZE,
            width=GRID_WIDTH,
            height=GRID_HEIGHT,
            wrap_longitude=True,
        )

    @staticmethod
    def _spatial_indices(
        lat: np.ndarray, lon: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Legacy clamped indexes used by precipitation and nowcast."""

        row_f = (NORTH - lat) / PIXEL_SIZE
        col_f = (lon - WEST) / PIXEL_SIZE
        row_floor = np.floor(row_f)
        col_floor = np.floor(col_f)
        r0_raw = row_floor.astype(np.int32)
        c0_raw = col_floor.astype(np.int32)
        r0 = np.clip(r0_raw, 0, GRID_HEIGHT - 1)
        c0 = np.clip(c0_raw, 0, GRID_WIDTH - 1)
        r1 = np.clip(r0_raw + 1, 0, GRID_HEIGHT - 1)
        c1 = np.clip(c0_raw + 1, 0, GRID_WIDTH - 1)
        dr = np.clip(row_f - row_floor, 0.0, 1.0).astype(np.float32)
        dc = np.clip(col_f - col_floor, 0.0, 1.0).astype(np.float32)
        return r0, c0, r1, c1, dr, dc

    def sampling_plan(
        self,
        z: int,
        x: int,
        y: int,
        tile_size: int = 256,
        padding: int = 0,
    ) -> SamplingPlan:
        """Return reusable geometry for an XYZ tile on the IFS grid."""

        return cached_regular_tile_sampling_plan(
            self.sampling_grid_identity,
            self._grid_version,
            z,
            x,
            y,
            tile_size,
            padding,
            "regular_latlon",
            WEST,
            NORTH,
            PIXEL_SIZE,
            PIXEL_SIZE,
            GRID_WIDTH,
            GRID_HEIGHT,
            True,
        )

    def _sample_physical_frame(
        self,
        frame: WeatherFrame,
        field: WeatherField,
        plan: SamplingPlan,
        bilinear: bool,
    ) -> np.ndarray:
        encoded = frame.field(field)
        if not bilinear:
            result = decode_field(field, encoded[plan.r0, plan.c0])
            return np.where(plan.valid, result, np.nan).astype(
                np.float32, copy=False
            )

        samples = (
            encoded[plan.r0, plan.c0],
            encoded[plan.r0, plan.c1],
            encoded[plan.r1, plan.c0],
            encoded[plan.r1, plan.c1],
        )
        spec = field_spec(field)
        has_nodata = spec.nodata is not None and any(
            np.any(sample == spec.nodata) for sample in samples
        )

        if not has_nodata:
            # Interpolation commutes with the field's affine decode. Work in
            # encoded space and reuse two float32 buffers instead of stacking
            # four decoded values, four weights, and four validity masks.
            top = samples[0].astype(np.float32)
            scratch = samples[1].astype(np.float32)
            scratch -= top
            scratch *= plan.dc
            top += scratch

            bottom = samples[2].astype(np.float32)
            scratch = samples[3].astype(np.float32)
            scratch -= bottom
            scratch *= plan.dc
            bottom += scratch
            bottom -= top
            bottom *= plan.dr
            top += bottom
            top *= np.float32(spec.scale)
            top += np.float32(spec.offset)
            top[~plan.valid] = np.nan
            return top

        # Rare nodata path: accumulate only valid neighbours without stack or
        # full-size np.where temporaries. The weighted encoded mean can still
        # be decoded once because every field codec is affine.
        weighted = np.zeros(plan.shape, dtype=np.float32)
        weight_sum = np.zeros(plan.shape, dtype=np.float32)
        scratch = np.empty(plan.shape, dtype=np.float32)
        one_minus_dr = np.subtract(np.float32(1.0), plan.dr)
        one_minus_dc = np.subtract(np.float32(1.0), plan.dc)
        weights = (
            one_minus_dr * one_minus_dc,
            one_minus_dr * plan.dc,
            plan.dr * one_minus_dc,
            plan.dr * plan.dc,
        )
        for sample, weight in zip(samples, weights, strict=True):
            valid = sample != spec.nodata
            scratch.fill(0.0)
            np.multiply(sample, weight, out=scratch, where=valid)
            weighted += scratch
            np.add(weight_sum, weight, out=weight_sum, where=valid)
        result = np.full(plan.shape, np.nan, dtype=np.float32)
        valid_result = (weight_sum > 0.0) & plan.valid
        np.divide(weighted, weight_sum, out=result, where=valid_result)
        result *= np.float32(spec.scale)
        result[valid_result] += np.float32(spec.offset)
        return result

    def _sample_field_with_plan(
        self,
        field: WeatherField,
        plan: SamplingPlan,
        timestamp: int | None,
        bilinear: bool,
    ) -> np.ndarray:
        """Sample native frames, interpolate requested points, then derive."""

        spec = field_spec(field)
        if spec.derived:
            dependencies = [
                self._sample_field_with_plan(
                    dependency, plan, timestamp, bilinear
                )
                for dependency in spec.dependencies
            ]
            if field is WeatherField.RELATIVE_HUMIDITY_2M:
                return relative_humidity_from_temperature_dewpoint(*dependencies)
            if field is WeatherField.WIND_SPEED_10M:
                return wind_speed_from_uv(*dependencies)
            raise ValueError(f"No derivation function for {field.value}")

        if field not in REQUIRED_WEATHER_FIELDS:
            raise KeyError(f"{self.name} does not provide {field.value}")

        frames = self._timesteps
        timestamps = self._field_timestamps(field, frames)
        if not timestamps:
            return np.full(plan.shape, np.nan, dtype=np.float32)
        if timestamp is None:
            timestamp = timestamps[-1]
        idx = int(np.searchsorted(timestamps, timestamp))
        if idx == 0:
            return self._sample_physical_frame(
                frames[timestamps[0]], field, plan, bilinear
            )
        if idx >= len(timestamps):
            return self._sample_physical_frame(
                frames[timestamps[-1]], field, plan, bilinear
            )
        before = timestamps[idx - 1]
        after = timestamps[idx]
        if timestamp == after:
            return self._sample_physical_frame(
                frames[after], field, plan, bilinear
            )
        before_values = self._sample_physical_frame(
            frames[before], field, plan, bilinear
        )
        after_values = self._sample_physical_frame(
            frames[after], field, plan, bilinear
        )
        fraction = np.float32((timestamp - before) / (after - before))
        if np.isfinite(before_values).all() and np.isfinite(after_values).all():
            after_values -= before_values
            after_values *= fraction
            before_values += after_values
            return before_values
        valid_before = np.isfinite(before_values)
        valid_after = np.isfinite(after_values)
        result = np.full(plan.shape, np.nan, dtype=np.float32)
        both = valid_before & valid_after
        result[both] = before_values[both] + fraction * (
            after_values[both] - before_values[both]
        )
        result[valid_before & ~valid_after] = before_values[valid_before & ~valid_after]
        result[valid_after & ~valid_before] = after_values[valid_after & ~valid_before]
        return result

    def sample_field(
        self,
        field: WeatherField,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = True,
    ) -> np.ndarray:
        normalized = WeatherField(field)
        if lat.shape != lon.shape:
            raise ValueError("lat and lon must have identical shapes")
        if normalized is WeatherField.PRECIPITATION:
            return decode_field(normalized, self.sample(lat, lon, timestamp, bilinear))
        plan = self._build_sampling_plan(lat, lon)
        return self._sample_field_with_plan(
            normalized, plan, timestamp, bilinear
        )

    def sample_tile_field(
        self,
        field: WeatherField,
        z: int,
        x: int,
        y: int,
        timestamp: int | None = None,
        tile_size: int = 256,
        padding: int = 0,
        bilinear: bool = True,
    ) -> np.ndarray:
        """Sample one tile without materialising an interpolated global frame."""

        normalized = WeatherField(field)
        plan = self.sampling_plan(z, x, y, tile_size, padding)
        if normalized is WeatherField.PRECIPITATION:
            # Precipitation retains its legacy nearest-time semantics. Generic
            # continuous weather layers use the optimized plan path above.
            lat, lon = web_mercator_tile_latlons(z, x, y, tile_size, padding)
            return decode_field(
                normalized,
                self.sample(lat, lon, timestamp, bilinear),
            )
        return self._sample_field_with_plan(
            normalized, plan, timestamp, bilinear
        )

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = False,
    ) -> np.ndarray:
        """Return legacy uint8 dBZ precipitation using nearest valid time."""

        frames = self._timesteps
        ts = self._nearest_from(self._precip_timestamps(frames), timestamp)
        if ts is None:
            return np.zeros(lat.shape, dtype=np.uint8)
        precip_dbz = frames[ts].field(WeatherField.PRECIPITATION)
        if not bilinear:
            row = ((NORTH - lat) / PIXEL_SIZE).astype(np.int32)
            col = ((lon - WEST) / PIXEL_SIZE).astype(np.int32)
            row = np.clip(row, 0, GRID_HEIGHT - 1)
            col = np.clip(col, 0, GRID_WIDTH - 1)
            return precip_dbz[row, col]

        r0, c0, r1, c1, dr, dc = self._spatial_indices(lat, lon)
        v00 = precip_dbz[r0, c0].astype(np.float32)
        v01 = precip_dbz[r0, c1].astype(np.float32)
        v10 = precip_dbz[r1, c0].astype(np.float32)
        v11 = precip_dbz[r1, c1].astype(np.float32)
        any_zero = (v00 == 0) | (v01 == 0) | (v10 == 0) | (v11 == 0)
        interp = (
            v00 * (1 - dr) * (1 - dc)
            + v01 * (1 - dr) * dc
            + v10 * dr * (1 - dc)
            + v11 * dr * dc
        )
        result = np.where(any_zero, v00, interp)
        return np.clip(result + 0.5, 0, 255).astype(np.uint8)

    @property
    def supports_snow(self) -> bool:
        return True

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        frames = self._timesteps
        ts = self._nearest_from(self._precip_timestamps(frames), timestamp)
        if ts is None:
            return np.zeros(lat.shape, dtype=bool)
        snow_mask = frames[ts].snow_mask
        if snow_mask is None:
            return np.zeros(lat.shape, dtype=bool)
        row = np.clip(
            ((NORTH - lat) / PIXEL_SIZE).astype(np.int32), 0, GRID_HEIGHT - 1
        )
        col = np.clip(
            ((lon - WEST) / PIXEL_SIZE).astype(np.int32), 0, GRID_WIDTH - 1
        )
        return snow_mask[row, col]

    def domain_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        if lat.shape != lon.shape:
            raise ValueError("lat and lon must have identical shapes")
        return (
            np.isfinite(lat)
            & np.isfinite(lon)
            & (lat >= -90.0)
            & (lat <= 90.0)
        )

    def feather_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return self.domain_mask(lat, lon).astype(np.float32)

    def has_data_at(self, timestamp: int) -> bool:
        return self._nearest_timestamp(timestamp) is not None

    def has_field_at(
        self,
        field: WeatherField,
        timestamp: int | None,
    ) -> bool:
        """Whether the last complete run can answer this field and valid time.

        Continuous IFS fields deliberately clamp outside their available time
        range, so a stale but complete run remains the global catch-all while a
        newer run is unavailable.
        """

        normalized = WeatherField(field)
        if normalized is WeatherField.PRECIPITATION:
            return bool(self._precip_timestamps())
        spec = field_spec(normalized)
        if spec.derived:
            return all(
                bool(self._field_timestamps(dependency))
                for dependency in spec.dependencies
            )
        return bool(self._field_timestamps(normalized))

    def has_data(self) -> bool:
        return bool(self._precip_timestamps())

    def health_status(self, now: int | None = None) -> dict:
        now = int(time.time()) if now is None else now
        valid_times = self.available_timestamps()
        loaded_fields = sorted(
            {field.value for frame in self._timesteps.values() for field in frame.fields}
        )
        run_age = None
        if self._reference_time:
            try:
                run_age = max(0, now - self._vt_to_unix(self._reference_time))
            except ValueError:
                pass
        stale = (
            not valid_times
            or valid_times[-1] < now
            or (run_age is not None and run_age > 12 * 3600)
        )
        return {
            "active_model_run": self._reference_time,
            "previous_model_run": self._previous_reference_time,
            "model_version": self.model_version,
            "valid_times": len(valid_times),
            "oldest_valid_time": valid_times[0] if valid_times else None,
            "latest_valid_time": valid_times[-1] if valid_times else None,
            "loaded_fields": loaded_fields,
            "field_bytes": self.field_bytes,
            "run_age_seconds": run_age,
            "stale": stale,
            "last_check_time": self._last_check_time,
            "last_successful_update": self._last_successful_update,
            "weather_fields_enabled": bool(self._weather_fields),
            "configured_fields": sorted(field.value for field in self._weather_fields),
            "last_update_error": self._last_update_error,
        }

    def _descriptor(self, values: np.ndarray) -> list:
        path = Path(str(values.filename))
        try:
            filename = path.relative_to(self._memmap_dir).as_posix()
        except ValueError:
            filename = path.name
        return [
            filename,
            values.dtype.str,
            list(values.shape),
        ]

    def __getstate__(self) -> dict:
        return self._state_for_frames(
            self._timesteps,
            reference_time=self._reference_time,
            previous_reference_time=self._previous_reference_time,
            last_modified_time=self._last_modified_time,
            last_successful_update=self._last_successful_update,
            last_update_error=self._last_update_error,
            content_version=self._content_version,
        )

    @staticmethod
    def _reopen_descriptor(memmap_dir: Path, descriptor: list) -> np.memmap:
        basename, dtype, shape = descriptor
        return np.memmap(
            memmap_dir / basename,
            dtype=np.dtype(dtype),
            mode="r",
            shape=tuple(shape),
        )

    def __setstate__(self, state: dict) -> None:
        """Restore v2 named frames or a legacy tuple-shaped snapshot."""

        memmap_dir = Path(state.get("memmap_dir", self._memmap_dir))
        new_timesteps: dict[int, WeatherFrame] = {}
        for ts_str, payload in state.get("timesteps", {}).items():
            ts = int(ts_str)
            if "fields" in payload:
                fields = {
                    WeatherField(field_name): self._reopen_descriptor(
                        memmap_dir, descriptor
                    )
                    for field_name, descriptor in payload["fields"].items()
                }
                snow_desc = payload.get("snow_mask")
                snow = (
                    self._reopen_descriptor(memmap_dir, snow_desc)
                    if snow_desc is not None
                    else None
                )
                new_timesteps[ts] = WeatherFrame(ts, fields, snow)
            else:
                # v1 snapshots stored implicit (precip, snow) descriptors.
                precip = self._reopen_descriptor(memmap_dir, payload["precip"])
                snow = self._reopen_descriptor(memmap_dir, payload["snow"])
                new_timesteps[ts] = WeatherFrame(
                    ts,
                    {WeatherField.PRECIPITATION: precip},
                    snow,
                )

        state.get("_precip_bboxes", {})  # discarded legacy Tier-2 state
        self._memmap_dir = memmap_dir
        self._runs_dir = self._memmap_dir / RUN_DIRECTORY_NAME
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._memmap_dir / ACTIVE_MANIFEST_NAME
        restored = _WeatherFrameDict()
        for ts, frame in new_timesteps.items():
            restored[ts] = frame
        self._timesteps = restored
        self._grid_version = int(state.get("grid_version", GRID_GEOMETRY_VERSION))
        self._content_version = int(state.get("content_version", 0))
        self._reference_time = state.get("reference_time")
        self._previous_reference_time = state.get("previous_reference_time")
        self._last_modified_time = state.get("last_modified_time")
        self._last_successful_update = state.get("last_successful_update")
        self._last_check_time = state.get("last_check_time")
        self._last_update_error = state.get("last_update_error")
        self._fs = None
        self._persistent = True

    async def close(self) -> None:
        self._timesteps.clear()
        self._fs = None
        if self._persistent:
            logger.info("ECMWF memmaps retained on disk at %s", self._memmap_dir)
        else:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("ECMWF memmap directory cleaned up")
