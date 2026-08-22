# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from librewxr.tiles.motion_renderer import (
    MOTION_VECTOR_OFFSET,
    MOTION_VECTOR_SCALE,
    render_motion_tile,
)
from librewxr.tiles.renderer import TileGeometry


def _geometry(values: np.ndarray) -> TileGeometry:
    return TileGeometry(
        values=values,
        snow_mask=None,
        tile_size=values.shape[0],
        pad=0,
        blur_radius=0.0,
    )


def _padded_geometry(values: np.ndarray, pad: int) -> TileGeometry:
    return TileGeometry(
        values=values,
        snow_mask=None,
        tile_size=values.shape[0] - 2 * pad,
        pad=pad,
        blur_radius=pad / 3,
    )


def _decode(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    packed = (
        rgb[..., 0].astype(np.uint32) * 65536
        + rgb[..., 1].astype(np.uint32) * 256
        + rgb[..., 2].astype(np.uint32)
    )
    packed_x = packed >> 12
    packed_y = packed & 0xFFF
    valid = packed != 0
    dx = (packed_x.astype(np.float32) - MOTION_VECTOR_OFFSET) / MOTION_VECTOR_SCALE
    dy = (packed_y.astype(np.float32) - MOTION_VECTOR_OFFSET) / MOTION_VECTOR_SCALE
    return dx, dy, valid


def test_motion_tile_tracks_translated_echo() -> None:
    frame0 = np.zeros((96, 96), dtype=np.uint8)
    frame1 = np.zeros_like(frame0)
    cv2.circle(frame0, (42, 48), 14, 180, thickness=-1)
    cv2.circle(frame1, (48, 45), 14, 180, thickness=-1)

    dx, dy, valid = _decode(render_motion_tile(_geometry(frame0), _geometry(frame1)))
    core = valid & (frame0 > 0)
    assert np.median(dx[core]) == pytest.approx(6.0, abs=1.25)
    assert np.median(dy[core]) == pytest.approx(-3.0, abs=1.25)


def test_motion_tile_is_zero_for_clear_pair() -> None:
    clear = np.zeros((64, 64), dtype=np.uint8)
    _, _, valid = _decode(render_motion_tile(_geometry(clear), _geometry(clear)))
    assert not valid.any()


def test_motion_tile_uses_padding_but_returns_exact_tile_size() -> None:
    pad = 16
    frame0 = np.zeros((128, 128), dtype=np.uint8)
    frame1 = np.zeros_like(frame0)
    cv2.circle(frame0, (18, 64), 13, 180, thickness=-1)
    cv2.circle(frame1, (24, 62), 13, 180, thickness=-1)

    dx, dy, valid = _decode(render_motion_tile(
        _padded_geometry(frame0, pad),
        _padded_geometry(frame1, pad),
    ))
    assert dx.shape == (96, 96)
    core = valid & (frame0[pad:-pad, pad:-pad] > 0)
    assert np.median(dx[core]) == pytest.approx(6.0, abs=1.25)
    assert np.median(dy[core]) == pytest.approx(-2.0, abs=1.25)
