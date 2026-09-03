# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Prime freshly written memmap files into the host page cache.

In the multi-worker split, render workers memory-map the same frame
files the pipeline writes and page-fault them in while holding the GIL;
on a slow backing disk those cold faults can stall a worker past
uvicorn's healthcheck timeout and get it SIGKILLed.  After each fetch
cycle the pipeline issues posix_fadvise(WILLNEED) for every memmap file
render workers will open, so the host page cache (shared between the
pipeline and renderer containers) already holds the frames by the time
any worker touches them.
"""
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_RECORD_NAME = "primed_memmaps.json"
_COORD_RECORD_NAME = "coord_pagecache_prime.json"
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


def _boot_id() -> str:
    """Best-effort host boot identifier used to invalidate stale records."""
    try:
        return _BOOT_ID_PATH.read_text().strip()
    except OSError:
        return "unknown"


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Best-effort atomic JSON write for tiny page-cache state files."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True))
        os.replace(tmp, path)
    except OSError:
        pass


def coord_pagecache_prime_stats(cache_dir: str | Path) -> dict | None:
    """Return the last coordinate-store priming record, if available."""
    try:
        payload = json.loads((Path(cache_dir) / _COORD_RECORD_NAME).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    payload["current_boot"] = payload.get("boot_id") == _boot_id()
    return payload


def prime_coord_store(
    cache_dir: str | Path,
    min_interval_seconds: float = 1800.0,
) -> dict:
    """Advise the kernel to retain shared coordinate arrays in page cache.

    Unlike frame memmaps, coordinate entries are long-lived and may keep the
    same mtime across process and host restarts.  The persistent record is
    therefore keyed by the host boot id and expires periodically: a reboot or
    an elapsed interval re-advises every current ``coord/**/*.npy`` entry.
    ``posix_fadvise(WILLNEED)`` is non-blocking best-effort I/O; failures are
    counted and never break a pipeline cycle.
    """
    root = Path(cache_dir)
    record_path = root / _COORD_RECORD_NAME
    now = time.time()
    boot_id = _boot_id()
    try:
        previous = json.loads(record_path.read_text())
    except (OSError, ValueError):
        previous = {}
    if not isinstance(previous, dict):
        previous = {}

    last_run = previous.get("completed_at")
    if (
        previous.get("boot_id") == boot_id
        and isinstance(last_run, (int, float))
        and now - float(last_run) < max(0.0, min_interval_seconds)
    ):
        return {**previous, "skipped": "interval"}

    started = time.monotonic()
    files = 0
    bytes_ = 0
    errors = 0
    supported = hasattr(os, "posix_fadvise")
    coord_root = root / "coord"
    paths = coord_root.rglob("*.npy") if coord_root.is_dir() else ()
    for path in paths:
        try:
            st = path.stat()
            if not supported:
                continue
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(
                    fd,
                    0,
                    0,
                    getattr(os, "POSIX_FADV_WILLNEED", 3),
                )
            finally:
                os.close(fd)
            files += 1
            bytes_ += st.st_size
        except OSError:
            errors += 1

    payload = {
        "status": "ok" if supported and errors == 0 else (
            "partial" if supported else "unsupported"
        ),
        "boot_id": boot_id,
        "completed_at": now,
        "files": files,
        "bytes": bytes_,
        "errors": errors,
        "duration_ms": round((time.monotonic() - started) * 1000, 1),
    }
    _write_json_atomic(record_path, payload)
    return payload


def _collect_memmap_files(store_payload: dict) -> set[Path]:
    """All .dat paths referenced by one store's snapshot payload.

    Snapshot payloads list memmap arrays as [basename, dtype, shape]
    triples nested at varying depths, under one or more "memmap_dir"
    keys.  Pair each basename with every memmap_dir found inside the
    same store subtree and dedupe.  One deviation: satellite (GMGSI)
    stores snapshot cache_root + channel + plain timestamps instead of a
    memmap_dir — their frame files live under
    <cache_root>/gmgsi/<layout>/<channel>/ and are resolved separately.
    """
    found: set[Path] = set()

    def dat_basenames(node) -> list[str]:
        names: list[str] = []
        if isinstance(node, dict):
            for value in node.values():
                names.extend(dat_basenames(value))
        elif isinstance(node, (list, tuple)):
            if (
                len(node) >= 2
                and isinstance(node[0], str)
                and node[0].endswith(".dat")
            ):
                names.append(node[0])
            else:
                for value in node:
                    names.extend(dat_basenames(value))
        return names

    def find_dirs(node) -> None:
        if isinstance(node, dict):
            dir_value = node.get("memmap_dir")
            if dir_value:
                base = Path(dir_value)
                for basename in dat_basenames(node):
                    found.add(base / basename)
            # Satellite (GMGSI) stores deviate: the snapshot carries
            # cache_root + channel + plain timestamps (no memmap_dir), and
            # the memmap files sit at
            # <cache_root>/gmgsi/<layout_version>/<channel>/frame_<ts>.dat.
            # Glob the layout dirs so a layout-version bump can't silently
            # stop priming the frames.
            cache_root = node.get("cache_root")
            channel = node.get("channel")
            timestamps = node.get("timestamps")
            if cache_root and channel and isinstance(timestamps, list):
                for layout_dir in Path(cache_root).glob("gmgsi/*"):
                    channel_dir = layout_dir / str(channel)
                    if channel_dir.is_dir():
                        for ts in timestamps:
                            if isinstance(ts, int):
                                found.add(channel_dir / f"frame_{ts}.dat")
            for value in node.values():
                find_dirs(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                find_dirs(value)

    find_dirs(store_payload)
    return found


def prime_fresh_memmaps(snapshot_payload: dict, cache_dir: str | Path) -> int:
    """Advise the kernel to preload freshly written memmap files.

    Walks the master-state snapshot, finds every .dat memmap file render
    workers will open, and issues posix_fadvise(WILLNEED) for any file not
    yet primed at its current mtime. Priming state persists in
    <cache_dir>/primed_memmaps.json. Never raises; returns bytes primed.
    """
    record_path = Path(cache_dir) / _RECORD_NAME
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        record = {}
    if not hasattr(os, "posix_fadvise"):
        return 0
    total = 0
    for store_payload in snapshot_payload.get("stores", {}).values():
        if not isinstance(store_payload, dict):
            continue
        for path in _collect_memmap_files(store_payload):
            try:
                st = path.stat()
            except OSError:
                continue
            key = str(path)
            if record.get(key) == st.st_mtime_ns:
                continue
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(
                        fd,
                        0,
                        0,
                        getattr(os, "POSIX_FADV_WILLNEED", 3),
                    )
                finally:
                    os.close(fd)
                record[key] = st.st_mtime_ns
                total += st.st_size
            except OSError:
                continue
    try:
        tmp = record_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record))
        os.replace(tmp, record_path)
    except OSError:
        pass
    return total
