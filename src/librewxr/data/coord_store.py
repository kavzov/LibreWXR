# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Content-addressed, size-capped on-disk store for tile coordinate arrays.

The tile renderer derives, per (region, z, x, y, tile_size, pad), several
numpy arrays that are pure functions of static inputs: pixel index grids
(``region_pixel_indices*``), fractional coordinate grids
(``region_pixel_indices_fractional*``), and lat/lon grids
(``tile_pixel_latlons*``) from ``librewxr.tiles.coordinates``.  Region
definitions never change between fetch cycles, so every render worker in a
multi-worker deployment computes the same arrays over and over.

This store moves those arrays onto the shared cache volume as read-only
memmaps - the same pattern FrameStore / the coverage-mask persistence use.
Compute once, store once physically; every worker maps the same pages
through the OS page cache.  Entries are content-addressed: the file name is
a SHA-1 digest of a key that folds in a signature over every projection
input, so any change to the projection math or region geometry namespaces
all entries and the stale files become unreachable garbage that the pruner
removes first (oldest mtime).

This module is a pure storage class: no ``librewxr.config`` settings are
imported.  Callers pass ``cache_dir`` and ``enabled_regions``.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Sequence

import numpy as np

from librewxr.data.regions import REGIONS

logger = logging.getLogger(__name__)

# Bump when the on-disk layout or the manifest schema changes.  The
# projection math gets its own version below.
FORMAT_VERSION = 1
# Bump on ANY change to the projection math in tiles/coordinates.py
# (LAEA/tmerc/equirect paths, WGS84 constants, grid derivation) - it
# namespaces every stored entry.
ALGO_VERSION = 1

# Entry kinds, one per cached function in tiles/coordinates.py.
KIND_INDICES = "indices"                # region_pixel_indices
KIND_INDICES_PAD = "indices_pad"        # region_pixel_indices_padded
KIND_FRACTIONAL = "fractional"          # region_pixel_indices_fractional
KIND_FRACTIONAL_PAD = "fractional_pad"  # region_pixel_indices_fractional_padded
KIND_FRACTIONAL_MASKED = "fractional_masked"
KIND_FRACTIONAL_MASKED_PAD = "fractional_masked_pad"
KIND_LATLON = "latlon"                  # tile_pixel_latlons
KIND_LATLON_PAD = "latlon_pad"          # tile_pixel_latlons_padded

# Subdirectory of the cache dir holding the coordinate entries.
_COORD_DIRNAME = "coord"
# prune() sweeps orphan *.tmp files (crashed writer) older than this.
_TMP_TTL_SECONDS = 3600.0
# prune() drains until the remaining entries fit in this fraction of budget.
_PRUNE_TARGET_FRACTION = 0.9
# stats() on-disk scan TTL (time.monotonic).
_STATS_TTL_SECONDS = 60.0
_LOCK_NAME = ".budget.lock"
_USAGE_NAME = ".usage"
_RESERVATION_DIRNAME = ".reservations"

# RegionDef fields the projection math in tiles/coordinates.py reads.
# Signature over-inclusion is harmless (it only busts the cache more often);
# under-inclusion is a correctness bug, so every field touched by the LAEA /
# tmerc / equirect paths is listed: bbox (west/east/south/north), derived
# width/height, pixel_size (+ pixel_size_y, which fixes ``_ps_y``), the
# ``proj`` selector, the shared grid derivation fields (grid_x_min,
# grid_y_max, grid_scale, grid_width, grid_height), and the LAEA / tmerc
# parameter blocks consumed by ``_laea_pixel_coords`` / ``_tmerc_pixel_coords``.
# WGS84 constants live in coordinates.py and are covered by ALGO_VERSION.
_REGION_SIGNATURE_FIELDS = (
    "west", "east", "south", "north",
    "width", "height",
    "pixel_size", "pixel_size_y",
    "proj",
    "grid_x_min", "grid_y_max", "grid_scale",
    "grid_width", "grid_height",
    "laea_lat0", "laea_lon0", "laea_x0", "laea_y0",
    "tmerc_lat0", "tmerc_lon0", "tmerc_radius", "tmerc_k0",
)


class CoordStore:
    """Content-addressed cache of coordinate arrays under ``cache_dir``.

    Each entry is one ``.npy`` file (via ``np.lib.format.open_memmap``)
    under ``<cache_dir>/coord/<sha1[:2]>/<sha1>.npy``.  ``open`` returns a
    read-only memmap (non-writeable, preserving the immutability contract
    the callers rely on); ``publish`` writes atomically (pid-unique tmp +
    ``os.replace``) and is a no-op when the key already exists, so
    concurrent publishers converge on one file.  Corruption / truncation
    self-heals by unlinking and falling back to compute; transient OSErrors
    (EIO / EAGAIN / EMFILE) fall through to compute WITHOUT unlinking.
    """

    def __init__(
        self, cache_dir: Path, enabled_regions: list[str], budget_bytes: int = 0,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._enabled_regions = list(enabled_regions)
        self._budget_bytes = max(0, int(budget_bytes))
        self._signature: str | None = None
        self._hits = 0
        self._misses = 0
        self._publishes = 0
        self._async_skipped = 0
        self._manifest_written = False
        self._async_executor: ThreadPoolExecutor | None = None
        self._async_pending: set[Path] = set()
        self._async_lock = Lock()
        # Cached (entries, bytes) from a recursive on-disk scan; TTL below.
        self._scan: tuple[int, int] | None = None
        self._scan_cached_at: float | None = None

    @contextmanager
    def _budget_lock(self):
        """Serialize capacity decisions and publishes across all workers."""
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / _LOCK_NAME
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _usage_path(self) -> Path:
        return self.root / _USAGE_NAME

    def _scan_entries(self) -> list[tuple[Path, int, float]]:
        entries: list[tuple[Path, int, float]] = []
        if self.root.is_dir():
            for path in self.root.rglob("*.npy"):
                try:
                    st = path.stat()
                except OSError:
                    continue
                entries.append((path, st.st_size, st.st_mtime))
        return entries

    def _read_usage_locked(self) -> int:
        try:
            return max(0, int(self._usage_path().read_text()))
        except (OSError, ValueError):
            total = self._reconciled_usage_locked()
            self._write_usage_locked(total)
            return total

    def _write_usage_locked(self, value: int) -> None:
        path = self._usage_path()
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(str(max(0, int(value))))
        os.replace(tmp, path)

    def _remove_orphan_tmps_locked(
        self, *, stale_only: bool = False,
    ) -> tuple[int, int]:
        """Remove tmp files; optionally retain files younger than the TTL."""
        removed_bytes = 0
        removed_entries = 0
        now = time.time()
        if not self.root.is_dir():
            return removed_bytes, removed_entries
        for path in self.root.rglob("*.tmp"):
            try:
                stat = path.stat()
                if stale_only and now - stat.st_mtime <= _TMP_TTL_SECONDS:
                    continue
                size = stat.st_size
                path.unlink()
            except OSError:
                continue
            removed_bytes += size
            removed_entries += 1
        return removed_bytes, removed_entries

    def _reservation_path(self, entry_path: Path) -> Path:
        return self.root / _RESERVATION_DIRNAME / f"{entry_path.stem}.reserve"

    def _reservation_bytes_locked(self, *, clean_stale: bool = False) -> int:
        total = 0
        now = time.time()
        root = self.root / _RESERVATION_DIRNAME
        if not root.is_dir():
            return 0
        for path in root.glob("*.reserve"):
            try:
                stat = path.stat()
                if clean_stale and now - stat.st_mtime > _TMP_TTL_SECONDS:
                    path.unlink()
                    continue
                total += max(0, int(path.read_text()))
            except (OSError, ValueError):
                try:
                    path.unlink()
                except OSError:
                    pass
        return total

    def _reconciled_usage_locked(self) -> int:
        entries = sum(size for _path, size, _mtime in self._scan_entries())
        return entries + self._reservation_bytes_locked(clean_stale=True)

    def _prune_entries_locked(
        self, target_bytes: int, reserved_bytes: int = 0,
    ) -> tuple[int, int, int]:
        """Prune oldest .npy entries and return removed bytes/count/remaining."""
        entries = self._scan_entries()
        total = sum(size for _path, size, _mtime in entries)
        removed_bytes = 0
        removed_entries = 0
        entries.sort(key=lambda entry: entry[2])
        for path, size, _mtime in entries:
            if total + reserved_bytes <= target_bytes:
                break
            try:
                path.unlink()
            except OSError:
                continue
            total -= size
            removed_bytes += size
            removed_entries += 1
        remaining = total + reserved_bytes
        self._write_usage_locked(remaining)
        return removed_bytes, removed_entries, remaining

    @property
    def root(self) -> Path:
        return self._cache_dir / _COORD_DIRNAME

    @property
    def signature(self) -> str:
        """SHA-256 hex over format/algo versions + enabled RegionDefs.

        Memoized; every entry key folds this in, so any code or region
        change namespaces the whole store.
        """
        if self._signature is None:
            payload: dict = {
                "format_version": FORMAT_VERSION,
                "algo_version": ALGO_VERSION,
                "enabled_regions": sorted(self._enabled_regions),
                "regions": {},
            }
            for name in sorted(self._enabled_regions):
                region = REGIONS.get(name)
                if region is None:
                    continue
                payload["regions"][name] = {
                    field: getattr(region, field)
                    for field in _REGION_SIGNATURE_FIELDS
                }
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"),
            ).encode()
            self._signature = hashlib.sha256(encoded).hexdigest()
        return self._signature

    def entry_path(
        self, kind: str, region: str | None, z: int, x: int, y: int,
        tile_size: int, pad: int,
    ) -> Path:
        """Path for one content-addressed entry (does not touch the disk)."""
        key = (
            f"{self.signature}|{kind}|{region or ''}|{z}|{x}|{y}"
            f"|{tile_size}|{pad}"
        )
        digest = hashlib.sha1(key.encode()).hexdigest()
        return self.root / digest[:2] / f"{digest}.npy"

    def open(
        self, kind: str, region: str | None, z: int, x: int, y: int,
        tile_size: int, pad: int, expected_shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> np.ndarray | None:
        """Return a read-only memmap for the entry, or None on any miss.

        On a corrupt header, shape/dtype mismatch, or truncated file the
        entry is unlinked (self-heal; the caller falls back to compute and
        re-publishes).  Transient OSErrors return None WITHOUT unlinking.
        """
        path = self.entry_path(kind, region, z, x, y, tile_size, pad)
        if not path.is_file():
            self._misses += 1
            return None
        try:
            arr = np.lib.format.open_memmap(path, mode="r")
        except FileNotFoundError:
            # Raced with a concurrent prune/unlink - the entry is simply gone.
            self._misses += 1
            return None
        except OSError as exc:
            # EIO / EAGAIN / EMFILE etc. - transient, do not destroy the file.
            logger.warning(
                "coord_store: cannot open %s (%s); leaving in place",
                path, exc,
            )
            self._misses += 1
            return None
        except ValueError:
            logger.warning("coord_store: corrupt header in %s; unlinking", path)
            path.unlink(missing_ok=True)
            self._misses += 1
            return None

        if tuple(arr.shape) != tuple(expected_shape) or arr.dtype != np.dtype(dtype):
            logger.warning(
                "coord_store: %s has shape %s dtype %s; expected %s %s; unlinking",
                path, arr.shape, arr.dtype, tuple(expected_shape),
                np.dtype(dtype),
            )
            del arr
            path.unlink(missing_ok=True)
            self._misses += 1
            return None

        try:
            st_size = path.stat().st_size
        except FileNotFoundError:
            del arr
            self._misses += 1
            return None
        if st_size < arr.nbytes:
            # Truncated write (crashed mid-publish): the declared data
            # extent does not fit.  Unlink; the caller recomputes.
            logger.warning(
                "coord_store: %s truncated (%d bytes < %d declared); unlinking",
                path, st_size, arr.nbytes,
            )
            del arr
            path.unlink(missing_ok=True)
            self._misses += 1
            return None
        if st_size > arr.nbytes:
            # .npy files legitimately carry a header on top of the data
            # extent, so excess is tolerable (log at debug only).
            logger.debug(
                "coord_store: %s larger than declared extent (%d > %d bytes)",
                path, st_size, arr.nbytes,
            )
        self._hits += 1
        return arr

    def publish(
        self, kind: str, region: str | None, z: int, x: int, y: int,
        tile_size: int, pad: int,
        data: np.ndarray | Sequence[np.ndarray],
    ) -> bool:
        """Persist ``data`` for the key; True when a new file was written.

        Content-addressed: if the final path already exists this is a
        no-op returning False (concurrent publishers write byte-identical
        content and converge on the first ``os.replace``).  Writes via a
        pid-unique tmp + ``os.replace`` so readers only ever see a complete
        file.  Best-effort: any failure is logged and returns False, never
        raises.  The first successful publish of the process also writes
        the manifest.
        """
        if isinstance(data, np.ndarray):
            shape = data.shape
            dtype = data.dtype
            parts = None
            data_bytes = data.nbytes
        else:
            parts = tuple(data)
            if not parts:
                return False
            first = np.asarray(parts[0])
            if any(
                np.asarray(part).shape != first.shape
                or np.asarray(part).dtype != first.dtype
                for part in parts[1:]
            ):
                logger.warning("coord_store: publish parts have mismatched layout")
                return False
            shape = (len(parts), *first.shape)
            dtype = first.dtype
            data_bytes = sum(np.asarray(part).nbytes for part in parts)

        # The .npy header is normally 128 bytes.  A 4 KiB allowance keeps
        # the physical store below the configured byte budget without
        # depending on NumPy's private header-size implementation.
        projected_bytes = data_bytes + 4096
        path = self.entry_path(kind, region, z, x, y, tile_size, pad)
        reservation = self._reservation_path(path)
        tmp: Path | None = None
        try:
            if self._budget_bytes:
                with self._budget_lock():
                    if path.exists() or reservation.exists():
                        return False
                    self._remove_orphan_tmps_locked(stale_only=True)
                    reserved_bytes = self._reservation_bytes_locked(
                        clean_stale=True,
                    )
                    usage = self._read_usage_locked()
                    if projected_bytes > self._budget_bytes:
                        logger.warning(
                            "coord_store: entry %s (%d bytes) exceeds %d-byte budget",
                            path, projected_bytes, self._budget_bytes,
                        )
                        return False
                    target = max(
                        0,
                        int(self._budget_bytes * _PRUNE_TARGET_FRACTION)
                        - projected_bytes,
                    )
                    if usage + projected_bytes > self._budget_bytes:
                        _removed_b, _removed_n, usage = self._prune_entries_locked(
                            target, reserved_bytes,
                        )
                    reservation.parent.mkdir(parents=True, exist_ok=True)
                    reservation.write_text(str(projected_bytes))
                    self._write_usage_locked(usage + projected_bytes)
            elif path.exists():
                return False

            # The expensive allocation/copy/flush is outside the capacity
            # lock. Different cold coordinates can therefore publish in
            # parallel while their conservative reservations keep the hard
            # byte limit intact.
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(
                f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            mm = np.lib.format.open_memmap(
                tmp, mode="w+", dtype=dtype, shape=shape,
            )
            try:
                if parts is None:
                    mm[:] = data
                else:
                    for index, part in enumerate(parts):
                        mm[index] = part
                mm.flush()
            finally:
                del mm
            actual_bytes = tmp.stat().st_size

            if self._budget_bytes:
                with self._budget_lock():
                    usage = self._read_usage_locked()
                    if usage - projected_bytes + actual_bytes > self._budget_bytes:
                        # Defensive only: projected_bytes includes a 4 KiB
                        # header allowance, so normal .npy output cannot land
                        # here.
                        tmp.unlink(missing_ok=True)
                        reservation.unlink(missing_ok=True)
                        self._write_usage_locked(max(0, usage - projected_bytes))
                        return False
                    if path.exists():
                        tmp.unlink(missing_ok=True)
                        reservation.unlink(missing_ok=True)
                        self._write_usage_locked(max(0, usage - projected_bytes))
                        return False
                    os.replace(tmp, path)
                    reservation.unlink(missing_ok=True)
                    self._write_usage_locked(
                        max(0, usage - projected_bytes + actual_bytes),
                    )
            else:
                if path.exists():
                    tmp.unlink(missing_ok=True)
                    return False
                os.replace(tmp, path)
        except Exception:
            logger.warning("coord_store: failed to publish %s", path, exc_info=True)
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            if self._budget_bytes:
                try:
                    with self._budget_lock():
                        reservation.unlink(missing_ok=True)
                        self._write_usage_locked(self._reconciled_usage_locked())
                except OSError:
                    pass
            return False
        self._publishes += 1
        self._scan = None
        if not self._manifest_written:
            self._manifest_written = True
            self.write_manifest()
        return True

    def publish_async(
        self, kind: str, region: str | None, z: int, x: int, y: int,
        tile_size: int, pad: int,
        data: np.ndarray | Sequence[np.ndarray],
        *, max_pending: int = 8,
    ) -> bool:
        """Schedule an atomic publish without delaying the render request.

        At most ``max_pending`` distinct entries are retained by the single
        writer thread in each process.  A full queue simply skips the shared
        publish: the caller still owns the freshly computed arrays in its
        in-process LRU, so correctness never depends on this optimisation.
        Duplicate keys in the same process are coalesced here; inter-process
        races still converge atomically in :meth:`publish`.
        """
        path = self.entry_path(kind, region, z, x, y, tile_size, pad)
        with self._async_lock:
            if path.exists() or path in self._async_pending:
                return False
            if len(self._async_pending) >= max(1, int(max_pending)):
                self._async_skipped += 1
                return False
            if self._async_executor is None:
                self._async_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="coord-store-writer",
                )
            self._async_pending.add(path)
            executor = self._async_executor

        def _write() -> None:
            try:
                self.publish(kind, region, z, x, y, tile_size, pad, data)
            finally:
                with self._async_lock:
                    self._async_pending.discard(path)

        try:
            executor.submit(_write)
        except RuntimeError:
            with self._async_lock:
                self._async_pending.discard(path)
                self._async_skipped += 1
            return False
        return True

    def close(self) -> None:
        """Finish queued asynchronous publishes and stop the writer."""
        with self._async_lock:
            executor = self._async_executor
            self._async_executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def prune(self, budget_bytes: int) -> tuple[int, int]:
        """Evict entries until the remainder fits ``budget_bytes``.

        First sweeps orphan ``*.tmp`` files older than one hour (crashed
        publishers); those do not count against the budget.  If the
        remaining entries exceed the budget, unlinks oldest-mtime-first
        until they fit 90% of it, then removes now-empty shard dirs.
        Returns ``(removed_bytes, removed_entries)`` including the tmp
        sweep.  Per-file failures are swallowed and eviction continues.
        """
        with self._budget_lock():
            removed_bytes, removed_entries = self._remove_orphan_tmps_locked(
                stale_only=True,
            )
            reserved_bytes = self._reservation_bytes_locked(clean_stale=True)
            entries = self._scan_entries()
            total = sum(size for _path, size, _mtime in entries) + reserved_bytes
            if total > budget_bytes:
                target = int(budget_bytes * _PRUNE_TARGET_FRACTION)
                pruned_bytes, pruned_entries, total = self._prune_entries_locked(
                    target, reserved_bytes,
                )
                removed_bytes += pruned_bytes
                removed_entries += pruned_entries
            else:
                self._write_usage_locked(total)

        # Remove now-empty shard dirs (rmdir only succeeds on empty dirs;
        # the manifest lives at root level and keeps root itself alive).
        if self.root.is_dir():
            for dirpath, _dirnames, _filenames in os.walk(
                self.root, topdown=False,
            ):
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass
        self._scan = None
        return removed_bytes, removed_entries

    def stats(self) -> dict:
        """Per-instance hit/miss/publish counters + on-disk size snapshot.

        The ``entries`` / ``bytes`` pair comes from a recursive scan of
        the store root, cached for 60s (``time.monotonic`` TTL) and
        invalidated by any publish/prune.
        """
        now = time.monotonic()
        if (
            self._scan is None
            or self._scan_cached_at is None
            or now - self._scan_cached_at >= _STATS_TTL_SECONDS
        ):
            entries = 0
            bytes_ = 0
            for _path, size, _mtime in self._scan_entries():
                entries += 1
                bytes_ += size
            self._scan = (entries, bytes_)
            self._scan_cached_at = now
        return {
            "hits": self._hits,
            "misses": self._misses,
            "publishes": self._publishes,
            "async_pending": len(self._async_pending),
            "async_skipped": self._async_skipped,
            "entries": self._scan[0],
            "bytes": self._scan[1],
            "budget_bytes": self._budget_bytes,
            "over_budget": bool(
                self._budget_bytes and self._scan[1] > self._budget_bytes
            ),
        }

    def write_manifest(self) -> None:
        """Atomically write the store manifest.  Best-effort, never raises."""
        try:
            payload = {
                "format_version": FORMAT_VERSION,
                "algo_version": ALGO_VERSION,
                "signature": self.signature,
                "written_at": time.time(),
                "pid": os.getpid(),
            }
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / (
                f"manifest.json.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            tmp.write_text(json.dumps(payload, sort_keys=True))
            os.replace(tmp, self.root / "manifest.json")
        except Exception:
            logger.warning("coord_store: failed to write manifest", exc_info=True)
