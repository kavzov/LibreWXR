# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""/health tile-cache breakdown classifies the fork's cache key shapes."""

import pytest

from librewxr.api.routes import _PRESENT_KEY_LEN, _tile_cache_kind

pytestmark = pytest.mark.api

TS = 1_790_000_000
PRESENT = (TS, 6, 33, 21, 256, True, False, 8, "png", 100, None)


def test_present_key_length_matches_radar_tile():
    assert len(PRESENT) == _PRESENT_KEY_LEN


@pytest.mark.parametrize("key, kind", [
    (("sat", TS, 3, 1, 2), "sat"),
    ((TS, 6, 33, 21, 256, True, False), "geometry"),
    (PRESENT, "present"),
    (PRESENT[:-1] + (15.0,), "present"),  # with a min_dbz threshold
    (PRESENT + ("light", "", 4, 0, "dots", "3-1790000000"), "overlay"),
    ((TS, "win", 6, 1024, 768, 256, True, False, 8, "png", 100), "window"),
    (("cov", TS, "win", 6, 1024, 768, 256), None),
    ((), None),
])
def test_tile_cache_kind(key, kind):
    assert _tile_cache_kind(key) == kind
