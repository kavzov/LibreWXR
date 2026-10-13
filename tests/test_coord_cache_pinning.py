# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Coordinate caches pin only store-backed arrays while a shared store is active.

Heap arrays returned while an async publish is still queued (or was skipped)
used to stay in each worker's LRU until evicted, growing every render worker
by several GiB of anon memory.
"""
import time

import numpy as np
import pytest

from librewxr.tiles import coordinates as C

pytestmark = pytest.mark.tiles


def _memmap(tmp_path, name="m.dat"):
    mm = np.memmap(tmp_path / name, dtype=np.float32, mode="w+", shape=(2, 4, 4))
    mm[:] = 1.0
    mm.flush()
    return np.memmap(tmp_path / name, dtype=np.float32, mode="r", shape=(2, 4, 4))


def _counting(result_for):
    calls = []

    @C._store_aware_cache(maxsize=4)
    def fn(key):
        calls.append(key)
        return result_for(key)

    return fn, calls


def test_without_store_caches_like_lru(monkeypatch):
    monkeypatch.setattr(C, "_get_store", lambda: None)
    fn, calls = _counting(lambda k: (np.zeros(3), np.ones(3)))
    a = fn(1)
    assert fn(1) is a
    assert calls == [1]
    for k in range(2, 7):
        fn(k)
    fn(1)  # evicted by maxsize=4 -> recomputed
    assert calls.count(1) == 2
    info = fn.cache_info()
    assert info.maxsize == 4 and info.currsize == 4 and fn.cache_held() == 0


def test_store_backed_result_is_pinned(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "_get_store", lambda: object())
    mm = _memmap(tmp_path)
    fn, calls = _counting(lambda k: (mm[0], mm[1]))
    first = fn("tile")
    assert fn("tile") is first
    assert calls == ["tile"]
    assert fn.cache_held() == 0


def test_heap_result_is_held_briefly_not_pinned(monkeypatch):
    monkeypatch.setattr(C, "_get_store", lambda: object())
    fn, calls = _counting(lambda k: (np.zeros(3), np.zeros(3)))
    first = fn("tile")
    assert fn("tile") is first  # burst on one tile: no recompute
    assert calls == ["tile"] and fn.cache_held() == 1

    monkeypatch.setattr(C, "_HEAP_HOLD_SECONDS", 0.0)
    fn.cache_clear()
    fn("tile")
    fn("tile")  # hold expired immediately -> recomputed, never pinned
    assert calls == ["tile", "tile", "tile"]
    assert fn.cache_held() <= 1


def test_heap_holds_are_bounded(monkeypatch):
    monkeypatch.setattr(C, "_get_store", lambda: object())
    monkeypatch.setattr(C, "_HEAP_HOLD_ENTRIES", 3)
    fn, _ = _counting(lambda k: np.zeros(3))
    for k in range(10):
        fn(k)
    assert fn.cache_held() == 3
    assert fn.cache_info().currsize == 3


def test_async_publish_upgrades_to_memmap(monkeypatch, tmp_path):
    """End to end: the first call holds a heap copy, a later call pins the
    published memmap and nothing heap-backed stays pinned."""
    from librewxr.config import settings

    monkeypatch.setattr(settings, "cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "coord_store_enabled", True)
    monkeypatch.setattr(settings, "coord_store_async_publish", True)
    C._reset_coord_store()
    C.tile_pixel_latlons_padded.cache_clear()
    try:
        first = C.tile_pixel_latlons_padded(5, 16, 10, 256, 8)
        assert not C._file_backed(first)
        store = C._get_store()
        deadline = time.monotonic() + 10
        while store.stats().get("entries", 0) < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        monkeypatch.setattr(C, "_HEAP_HOLD_SECONDS", 0.0)
        C.tile_pixel_latlons_padded.cache_clear()
        second = C.tile_pixel_latlons_padded(5, 16, 10, 256, 8)
        assert C._file_backed(second)
        assert np.array_equal(first[0], second[0])
        assert C.tile_pixel_latlons_padded(5, 16, 10, 256, 8) is second
        assert C.tile_pixel_latlons_padded.cache_held() == 0
    finally:
        C.tile_pixel_latlons_padded.cache_clear()
        C._reset_coord_store()
