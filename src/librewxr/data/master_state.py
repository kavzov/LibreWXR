# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Cross-process state snapshots for the multi-worker tile-server split.

The data pipeline publishes immutable generations under the shared cache
volume after each fetch cycle.  Every generation contains a manifest plus
hardlinks to the memmap files referenced by that manifest.  Only after the
generation is complete is the top-level ``state.json`` atomically replaced.
Render-only workers therefore see either the complete old generation or the
complete new one, never a manifest whose files were already evicted.

Format::

    {
      "version": 1,
      "written_at": 1712345600,
      "stores": {
        "frame_store":  { ... __getstate__ output ... },
        "ecmwf_grid":   { "format_version": 2, "timesteps": { ... } },
        "hrrr_grid":    { ... },
        ...
      }
    }

Stores whose value is ``None`` (disabled by config) are skipped.  Stores
present in the snapshot but absent from the consumer's ``stores`` dict
are silently ignored, so a tile-server worker can be started with a
subset of stores enabled.

The envelope stays at version 1 for compatibility. Individual stores may
version their own additive/representation changes; ECMWF IFS uses store format
2 and its ``__setstate__`` also accepts the legacy implicit precip/snow shape.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_FILENAME = "state.json"
STATE_VERSION = 1
STATE_GENERATIONS_DIRNAME = "state-generations"
DEFAULT_STATE_RETENTION_GENERATIONS = 3
_PATH_KEYS = frozenset({"memmap_dir", "cache_root"})
_DESCRIPTOR_STATE_KEYS = frozenset({
    "frames",
    "animation_frames",
    "flows",
    "nwp_flow",
    "masks",
    "cells",
    "timesteps",
})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably write *payload* to *path* (which must not be visible yet)."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, default=str)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes across a host crash."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_under(path: Path, root: Path) -> Path | None:
    """Return *path* relative to *root*, or ``None`` when it is external."""
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def _descriptor_paths(value: Any) -> set[Path]:
    """Find relative ``[filename, dtype, shape]`` memmap descriptors."""
    paths: set[Path] = set()
    if isinstance(value, dict):
        for child in value.values():
            paths.update(_descriptor_paths(child))
    elif isinstance(value, list):
        if (
            len(value) == 3
            and isinstance(value[0], str)
            and isinstance(value[2], list)
        ):
            candidate = Path(value[0])
            if not candidate.is_absolute() and ".." not in candidate.parts:
                paths.add(candidate)
        else:
            for child in value:
                paths.update(_descriptor_paths(child))
    return paths


def _snapshot_sources(
    value: Any,
    cache_dir: Path,
) -> dict[Path, set[Path] | None]:
    """Collect persistent roots and, where possible, exact files to retain."""
    sources: dict[Path, set[Path] | None] = {}
    if isinstance(value, dict):
        memmap_dir = value.get("memmap_dir")
        if isinstance(memmap_dir, str):
            root = Path(memmap_dir)
            if _path_under(root, cache_dir) is not None:
                files = (
                    _descriptor_paths(value)
                    if _DESCRIPTOR_STATE_KEYS.intersection(value)
                    else None
                )
                existing = sources.get(root, set())
                sources[root] = (
                    None if existing is None or files is None else existing | files
                )
        cache_root = value.get("cache_root")
        if isinstance(cache_root, str):
            root = Path(cache_root) / "gmgsi"
            if _path_under(root, cache_dir) is not None:
                sources[root] = None
        for child in value.values():
            for root, files in _snapshot_sources(child, cache_dir).items():
                existing = sources.get(root, set())
                sources[root] = (
                    None if existing is None or files is None else existing | files
                )
    elif isinstance(value, list):
        for child in value:
            for root, files in _snapshot_sources(child, cache_dir).items():
                existing = sources.get(root, set())
                sources[root] = (
                    None if existing is None or files is None else existing | files
                )
    return sources


def _hardlink_tree(source: Path, destination: Path) -> int:
    """Mirror regular files below *source* into *destination* as hardlinks."""
    destination.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return 0
    linked = 0
    for root, dirnames, filenames in os.walk(source):
        root_path = Path(root)
        target_root = destination / root_path.relative_to(source)
        target_root.mkdir(parents=True, exist_ok=True)
        dirnames[:] = [name for name in dirnames if not name.endswith(".tmp")]
        for filename in filenames:
            if filename.endswith(".tmp"):
                continue
            os.link(root_path / filename, target_root / filename)
            linked += 1
    return linked


def _hardlink_selected(
    source: Path,
    destination: Path,
    relative_paths: set[Path],
) -> int:
    """Hardlink only manifest-referenced files, preserving subdirectories."""
    destination.mkdir(parents=True, exist_ok=True)
    linked = 0
    for relative in sorted(relative_paths):
        if relative.name.endswith(".tmp"):
            continue
        src = source / relative
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        linked += 1
    return linked


def _rewrite_store_paths(value: Any, cache_dir: Path, snapshot_root: Path) -> Any:
    """Point store directory fields at the immutable generation mirror."""
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        for key, child in value.items():
            if key in _PATH_KEYS and isinstance(child, str):
                relative = _path_under(Path(child), cache_dir)
                if relative is not None:
                    rewritten[key] = str(snapshot_root / relative)
                    continue
            rewritten[key] = _rewrite_store_paths(child, cache_dir, snapshot_root)
        return rewritten
    if isinstance(value, list):
        return [_rewrite_store_paths(child, cache_dir, snapshot_root) for child in value]
    return value


def _prune_generations(generations_dir: Path, keep: int) -> None:
    """Remove complete generations older than the newest *keep* entries."""
    complete = sorted(
        path
        for path in generations_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    for path in complete[:-keep]:
        shutil.rmtree(path)


def snapshot_state(stores: dict[str, Any]) -> dict:
    """Build the ``state.json`` payload dict from a mapping of stores.

    Extracted from ``dump_state`` so the (fast) in-memory dict building
    half can run on the event loop while the (slower) JSON serialisation
    + atomic rename runs in a worker thread.

    Args:
        stores: mapping of store name → store object.  Values that are
            ``None`` are skipped.  Each non-None value must implement
            ``__getstate__()`` returning a JSON-serialisable dict.

    Returns:
        The payload dict, ready for :func:`write_state_snapshot`.
    """
    payload: dict[str, Any] = {
        "version": STATE_VERSION,
        "written_at": int(time.time()),
        "stores": {},
    }
    for name, obj in stores.items():
        if obj is None:
            continue
        # Python 3.11+ adds a default ``object.__getstate__`` to every
        # class.  ``hasattr`` therefore can't distinguish stores that
        # explicitly opt in from arbitrary objects — but the default
        # returns ``None`` for objects with no instance state, so we
        # filter on the result instead.
        try:
            store_state = obj.__getstate__()
        except Exception:
            logger.exception("Failed to serialise store %r, skipping", name)
            continue
        if store_state is None:
            logger.warning(
                "Store %r returned no state, skipping in state.json", name,
            )
            continue
        payload["stores"][name] = store_state
    return payload


def write_state_snapshot(
    payload: dict,
    cache_dir: Path,
    retention_generations: int = DEFAULT_STATE_RETENTION_GENERATIONS,
) -> Path:
    """Publish an immutable, atomically-selected state generation.

    The caller may build ``payload`` on the event loop and run this slower
    hardlink/JSON/fsync phase in a worker thread.  The input mapping is not
    mutated, so it remains suitable for page-cache priming afterwards.

    Args:
        payload: dict built by :func:`snapshot_state`.
        cache_dir: directory shared with the tile-server workers.
            ``state.json`` is written at the top level of this directory.
        retention_generations: complete generations to retain, including the
            current one.  At least two keep racing readers safe.

    Returns:
        The path to the newly-written ``state.json``.
    """
    if retention_generations < 2:
        raise ValueError("retention_generations must be at least 2")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    generations_dir = cache_dir / STATE_GENERATIONS_DIRNAME
    generations_dir.mkdir(parents=True, exist_ok=True)
    for stale in generations_dir.glob(".*.tmp"):
        if stale.is_dir():
            shutil.rmtree(stale)

    generation_id = f"{time.time_ns():020d}-{os.getpid()}"
    staging = generations_dir / f".{generation_id}.tmp"
    generation_dir = generations_dir / generation_id
    staging_files = staging / "files"
    final_files = generation_dir / "files"
    staging_files.mkdir(parents=True)

    linked = 0
    published_payload = dict(payload)
    published_payload["stores"] = dict(payload.get("stores", {}))
    try:
        for source_root, relative_paths in _snapshot_sources(
            published_payload, cache_dir
        ).items():
            relative = _path_under(source_root, cache_dir)
            if relative is None:
                continue
            destination = staging_files / relative
            if relative_paths is None:
                linked += _hardlink_tree(source_root, destination)
            else:
                linked += _hardlink_selected(source_root, destination, relative_paths)

        published_payload["stores"] = _rewrite_store_paths(
            published_payload["stores"], cache_dir, final_files
        )
        published_payload["generation"] = {
            "id": generation_id,
            "path": str(generation_dir),
            "retention_generations": retention_generations,
        }
        _write_json(staging / STATE_FILENAME, published_payload)
        os.replace(staging, generation_dir)
        _fsync_directory(generations_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final = cache_dir / STATE_FILENAME
    tmp = cache_dir / f".{STATE_FILENAME}.tmp"
    _write_json(tmp, published_payload)
    os.replace(tmp, final)
    _fsync_directory(cache_dir)
    try:
        _prune_generations(generations_dir, retention_generations)
    except Exception:
        # Publication already succeeded.  Extra old generations consume disk
        # but cannot compromise readers, so leave cleanup for the next cycle.
        logger.exception("Failed to prune old state generations")
    logger.debug(
        "Published state generation %s: %d store(s), %d hardlink(s) → %s "
        "(%.2fs)",
        generation_id,
        len(published_payload["stores"]),
        linked,
        final,
        time.monotonic() - started,
    )
    return final


def dump_state(
    stores: dict[str, Any],
    cache_dir: Path,
    retention_generations: int = DEFAULT_STATE_RETENTION_GENERATIONS,
) -> Path:
    """Atomically write a snapshot of every store's ``__getstate__`` to disk.

    Composes the two halves of the pipeline's snapshot path —
    :func:`snapshot_state` builds the payload from the live stores and
    :func:`write_state_snapshot` writes it atomically — so callers that
    need the write off the event loop can invoke the halves separately.

    The write goes to ``<cache_dir>/.state.json.tmp`` first and is then
    atomically renamed to ``state.json``.  Concurrent readers either see
    the old file (if they read before the rename) or the new file
    (after) — never a partial write.

    Args:
        stores: mapping of store name → store object.  Values that are
            ``None`` are skipped.  Each non-None value must implement
            ``__getstate__()`` returning a JSON-serialisable dict.
        cache_dir: directory shared with the tile-server workers.
            ``state.json`` is written at the top level of this directory.

    Returns:
        The path to the newly-written ``state.json``.
    """
    return write_state_snapshot(
        snapshot_state(stores),
        cache_dir,
        retention_generations=retention_generations,
    )


def load_state(cache_dir: Path) -> dict[str, Any] | None:
    """Read ``state.json`` from ``cache_dir``.

    Returns the parsed payload (with ``version`` / ``written_at`` /
    ``stores`` keys) or ``None`` if the file is absent.  Raises if the
    file exists but is malformed — callers should let those propagate
    so the worker fails loudly on corruption.
    """
    path = Path(cache_dir) / STATE_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != STATE_VERSION:
        logger.warning(
            "state.json version mismatch: file has %r, expected %d",
            version, STATE_VERSION,
        )
    return payload


def apply_state(
    payload: dict[str, Any],
    stores: dict[str, Any],
    prev_payload: dict[str, Any] | None = None,
) -> list[str]:
    """Call ``__setstate__`` on every matching store in ``stores``.

    Args:
        payload: parsed result of :func:`load_state`.
        stores: mapping of store name → store object.  Stores whose
            value is ``None`` are skipped (e.g. disabled by config).
            Stores present in ``payload["stores"]`` but absent (or
            ``None``) here are silently ignored.
        prev_payload: the previously-applied parsed payload.  A store
            whose payload sub-dict is equal to the previous one is
            skipped entirely — on cycles with no new NWP model run every
            grid payload is identical, which eliminates most of the
            ~800 memmap re-opens per poll.  ``None`` (boot path, or a
            caller with no apply history) reloads every store.

    Returns:
        List of store names that were successfully refreshed.
    """
    refreshed: list[str] = []
    snapshot = payload.get("stores", {})
    prev_snapshot = prev_payload.get("stores", {}) if prev_payload else {}
    for name, obj in stores.items():
        if obj is None:
            continue
        store_state = snapshot.get(name)
        if store_state is None:
            continue
        # Incremental reload: skip stores whose payload sub-dict is
        # identical to the previously-applied snapshot.  Comparison is
        # on the parsed dicts (deep ==), not raw JSON strings.
        #
        # KNOWN EXCEPTION: alerts_store and storm_cell_store embed a
        # last_updated wall-clock in their payload, so they compare
        # unequal every cycle and always reload.  Accepted — they are
        # tiny.
        prev_state = prev_snapshot.get(name)
        if prev_snapshot and prev_state == store_state:
            continue
        try:
            obj.__setstate__(store_state)
            refreshed.append(name)
        except Exception:
            logger.exception("Failed to apply state for store %r", name)
    return refreshed


def state_mtime(cache_dir: Path) -> float | None:
    """Return the modification time of ``state.json`` (or ``None``).

    Used by tile-server workers to poll for changes without re-reading
    and re-parsing the file every tick.
    """
    path = Path(cache_dir) / STATE_FILENAME
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _load_and_apply_state(
    cache_dir: Path,
    stores: dict[str, object | None],
    prev_payload: dict | None,
) -> tuple[dict | None, list[str]]:
    """Read + diff + apply a state snapshot, off the event loop.

    Runs in a worker thread via :func:`asyncio.to_thread` so the
    ``json.loads`` and memmap re-open cost of ``load_state`` +
    ``apply_state`` never blocks the event loop.  Thread-safety basis
    for applying store state from a thread while renders run in other
    threads:
    (a) the GIL makes every reference assignment in ``__setstate__``
        atomic;
    (b) every ``__setstate__`` builds new structures and then swaps
        references — verified for FrameStore, NowcastStore,
        PrecipMaskStore, StormCellStore, AlertsStore, the IFS / HRRR /
        HRRR-Alaska / HRDPS / ICON-EU / DMI-DINI / JMA-MSM / WRF-SMN /
        AROME grids, and the GMGSI satellite source; old structures are
        never mutated in place;
    (c) in-flight renders hold references to the old structures, so the
        old memmap inodes stay alive until those renders release them.

    Returns ``(payload, refreshed)``; ``payload`` is ``None`` when the
    snapshot file vanished between the mtime check and the read.
    """
    payload = load_state(cache_dir)
    if payload is None:
        return None, []
    refreshed = apply_state(payload, stores, prev_payload=prev_payload)
    return payload, refreshed
