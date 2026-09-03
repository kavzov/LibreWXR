# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the pipeline page-cache priming (``data/pagecache.py``).

Cover the snapshot-payload walker (memmap_dir + basename pairing,
satellite-style cache_root/channel resolution, noise filtering, dedupe)
and the end-to-end prime path (mtime dedupe, corrupt-record recovery,
missing-file tolerance).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from librewxr.data import pagecache


def test_collect_memmap_files_frame_store():
    payload = {
        "memmap_dir": "/tmp/radar",
        "frames": [
            {
                "timestamp": 1,
                "regions": {"USCOMP": ["1_USCOMP.dat", "|u1", [4, 4]]},
            }
        ],
        "max_frames": 12,
    }
    assert pagecache._collect_memmap_files(payload) == {
        Path("/tmp/radar/1_USCOMP.dat"),
    }


def test_collect_memmap_files_nwp():
    payload = {
        "memmap_dir": "/tmp/ecmwf_ifs",
        "timesteps": {
            "1700000000": {
                "precip": ["1700000000_precip.dat", "|u1", [4, 4]],
                "snow": ["1700000000_snow.dat", "|u1", [4, 4]],
            }
        },
    }
    assert pagecache._collect_memmap_files(payload) == {
        Path("/tmp/ecmwf_ifs/1700000000_precip.dat"),
        Path("/tmp/ecmwf_ifs/1700000000_snow.dat"),
    }


def test_collect_memmap_files_ignores_non_memmap_and_dedupes():
    payload = {
        "memmap_dir": "/tmp/radar",
        "shapes": [4, 4],  # shape list: first element is not a str
        "names": ["USCOMP"],  # plain string, not a .dat basename
        "frames": [
            {
                "timestamp": 1,
                "regions": {
                    "USCOMP": ["1_USCOMP.dat", "|u1", [4, 4]],
                    "RRQPE": ["1_RRQPE.dat", "|u1", [4, 4]],
                },
            },
            {
                # Same basename again — must be deduped away.
                "timestamp": 2,
                "regions": {"USCOMP": ["1_USCOMP.dat", "|u1", [4, 4]]},
            },
        ],
        "max_frames": 12,
    }
    assert pagecache._collect_memmap_files(payload) == {
        Path("/tmp/radar/1_USCOMP.dat"),
        Path("/tmp/radar/1_RRQPE.dat"),
    }


def test_prime_fresh_memmaps_end_to_end(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    radar_dir = cache_dir / "radar"
    radar_dir.mkdir(parents=True)
    f1 = radar_dir / "1_USCOMP.dat"
    f2 = radar_dir / "2_USCOMP.dat"
    f1.write_bytes(b"x" * 16)
    f2.write_bytes(b"y" * 32)
    payload = {
        "stores": {
            "frame_store": {
                "memmap_dir": str(radar_dir),
                "frames": [
                    {"timestamp": 1, "regions": {"USCOMP": ["1_USCOMP.dat", "|u1", [4, 4]]}},
                    {"timestamp": 2, "regions": {"USCOMP": ["2_USCOMP.dat", "|u1", [4, 4]]}},
                ],
                "max_frames": 12,
            }
        }
    }
    opened: list[str] = []
    advised: list[int] = []
    fd_to_path: dict[int, str] = {}
    next_fd = iter(range(100, 200))

    def fake_open(path, flags):
        fd = next(next_fd)
        fd_to_path[fd] = str(path)
        opened.append(str(path))
        return fd

    def fake_fadvise(fd, offset, length, advice):
        advised.append(fd)

    monkeypatch.setattr(pagecache.os, "open", fake_open)
    monkeypatch.setattr(pagecache.os, "close", lambda fd: None)
    monkeypatch.setattr(pagecache.os, "posix_fadvise", fake_fadvise, raising=False)

    # First call: both files primed once, bytes primed = sum of sizes.
    assert pagecache.prime_fresh_memmaps(payload, cache_dir) == 16 + 32
    assert set(opened) == {str(f1), str(f2)}
    assert {fd_to_path[fd] for fd in advised} == {str(f1), str(f2)}
    assert (cache_dir / "primed_memmaps.json").exists()

    # Second call with unchanged files: nothing re-primed.
    advised.clear()
    assert pagecache.prime_fresh_memmaps(payload, cache_dir) == 0
    assert advised == []

    # Touch one file: only it gets re-primed (its own size).
    st = f1.stat()
    os.utime(f1, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    advised.clear()
    assert pagecache.prime_fresh_memmaps(payload, cache_dir) == 16
    assert {fd_to_path[fd] for fd in advised} == {str(f1)}


def test_prime_fresh_memmaps_missing_file_and_corrupt_record(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Corrupt the priming record file first.
    record = cache_dir / "primed_memmaps.json"
    record.write_text("{not valid json")
    payload = {
        "stores": {
            "frame_store": {
                "memmap_dir": str(cache_dir),
                "frames": [
                    {"timestamp": 1, "regions": {"USCOMP": ["1_USCOMP.dat", "|u1", [4, 4]]}},
                ],
                "max_frames": 12,
            }
        }
    }
    advised = []
    monkeypatch.setattr(
        pagecache.os,
        "posix_fadvise",
        lambda *args: advised.append(args),
        raising=False,
    )
    # The referenced file does not exist -> skipped, no raise.
    assert pagecache.prime_fresh_memmaps(payload, cache_dir) == 0
    assert advised == []
    # The corrupt record file is rewritten as valid JSON.
    assert json.loads(record.read_text()) == {}


def test_prime_fresh_memmaps_without_stores(tmp_path, monkeypatch):
    advised = []
    monkeypatch.setattr(
        pagecache.os,
        "posix_fadvise",
        lambda *args: advised.append(args),
        raising=False,
    )
    assert pagecache.prime_fresh_memmaps({"version": 1, "written_at": 0}, tmp_path) == 0
    assert advised == []


def test_prime_coord_store_repeats_after_interval_or_reboot(tmp_path, monkeypatch):
    coord_dir = tmp_path / "coord" / "ab"
    coord_dir.mkdir(parents=True)
    first = coord_dir / "first.npy"
    second = coord_dir / "second.npy"
    first.write_bytes(b"x" * 16)
    second.write_bytes(b"y" * 32)

    opened: list[str] = []
    next_fd = iter(range(200, 300))
    monkeypatch.setattr(pagecache, "_boot_id", lambda: "boot-a")
    monkeypatch.setattr(pagecache.os, "open", lambda path, flags: (
        opened.append(str(path)) or next(next_fd)
    ))
    monkeypatch.setattr(pagecache.os, "close", lambda fd: None)
    monkeypatch.setattr(pagecache.os, "posix_fadvise", lambda *args: None, raising=False)
    times = iter([1000.0, 10.0, 10.025, 1010.0, 1020.0, 20.0, 20.010])
    monkeypatch.setattr(pagecache.time, "time", lambda: next(times))
    monkeypatch.setattr(pagecache.time, "monotonic", lambda: next(times))

    first_run = pagecache.prime_coord_store(tmp_path, min_interval_seconds=30)
    assert first_run["files"] == 2
    assert first_run["bytes"] == 48
    assert first_run["errors"] == 0
    assert len(opened) == 2

    skipped = pagecache.prime_coord_store(tmp_path, min_interval_seconds=30)
    assert skipped["skipped"] == "interval"
    assert len(opened) == 2

    monkeypatch.setattr(pagecache, "_boot_id", lambda: "boot-b")
    rebooted = pagecache.prime_coord_store(tmp_path, min_interval_seconds=30)
    assert rebooted["files"] == 2
    assert len(opened) == 4
    stats = pagecache.coord_pagecache_prime_stats(tmp_path)
    assert stats["current_boot"] is True
    assert stats["boot_id"] == "boot-b"


def test_prime_coord_store_tolerates_missing_store(tmp_path, monkeypatch):
    monkeypatch.setattr(pagecache, "_boot_id", lambda: "boot-a")
    monkeypatch.setattr(pagecache.os, "posix_fadvise", lambda *args: None, raising=False)
    result = pagecache.prime_coord_store(tmp_path, min_interval_seconds=0)
    assert result["status"] == "ok"
    assert result["files"] == 0
    assert result["bytes"] == 0
    assert pagecache.coord_pagecache_prime_stats(tmp_path)["current_boot"] is True
