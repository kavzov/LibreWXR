# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Compact optical-flow tiles for client-side radar animation.

The public radar PNGs are presentation products, so a browser cannot infer
where their pixels should move between timestamps.  This module computes the
motion field from the same post-composite tile geometry the public renderer
uses and packs signed x/y displacement into an ordinary opaque RGB PNG:

* 12 high bits: x displacement
* 12 low bits: y displacement
* RGB 0/0/0: no valid motion at this pixel

Valid components use an offset of 2048 and half-pixel precision.  The field
therefore describes displacement over the complete requested timestamp pair,
not velocity per second.  A WebGL client can multiply it by an interpolation
fraction without needing any knowledge of the source radar projection.
"""

import io

import cv2
import numpy as np
from PIL import Image

from librewxr.tiles.renderer import TileGeometry


MOTION_ENCODING = "rgb12-offset-2048"
MOTION_VECTOR_OFFSET = 2048
MOTION_VECTOR_SCALE = 2.0
MOTION_RENDERER_VERSION = 1


def _geometry_values(geometry: TileGeometry) -> np.ndarray:
    """Return geometry including its overlap padding, or a clear tile."""
    if geometry.is_transparent:
        return np.zeros((geometry.tile_size, geometry.tile_size), dtype=np.uint8)
    return np.ascontiguousarray(geometry.values, dtype=np.uint8)


def _pack_motion(flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    encoded_x = np.clip(
        np.rint(flow[..., 0] * MOTION_VECTOR_SCALE) + MOTION_VECTOR_OFFSET,
        1,
        4095,
    ).astype(np.uint16)
    encoded_y = np.clip(
        np.rint(flow[..., 1] * MOTION_VECTOR_SCALE) + MOTION_VECTOR_OFFSET,
        1,
        4095,
    ).astype(np.uint16)
    packed = (encoded_x.astype(np.uint32) << 12) | encoded_y.astype(np.uint32)
    packed[~valid] = 0
    rgb = np.empty((*valid.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (packed >> 16).astype(np.uint8)
    rgb[..., 1] = ((packed >> 8) & 0xFF).astype(np.uint8)
    rgb[..., 2] = (packed & 0xFF).astype(np.uint8)
    return rgb


def render_motion_tile(previous: TileGeometry, following: TileGeometry) -> bytes:
    """Compute and encode optical flow between two rendered radar geometries."""
    if previous.tile_size != following.tile_size:
        raise ValueError("motion geometries must have equal tile sizes")

    frame0 = _geometry_values(previous)
    frame1 = _geometry_values(following)
    pad = previous.pad
    if frame0.shape != frame1.shape or previous.pad != following.pad:
        # Region availability can occasionally change at a frame boundary.
        # Fall back to equal tile-sized fields instead of rejecting the pair.
        frame0 = frame0[
            previous.pad:previous.pad + previous.tile_size,
            previous.pad:previous.pad + previous.tile_size,
        ]
        frame1 = frame1[
            following.pad:following.pad + following.tile_size,
            following.pad:following.pad + following.tile_size,
        ]
        pad = 0
    active = (frame0 > 0) | (frame1 > 0)

    rgb = np.zeros((*frame0.shape, 3), dtype=np.uint8)
    if active.any():
        # A small blur suppresses dBZ quantisation noise without moving storm
        # edges.  Farneback then estimates dense forward displacement A -> B.
        source = cv2.GaussianBlur(frame0.astype(np.float32), (0, 0), 0.8)
        target = cv2.GaussianBlur(frame1.astype(np.float32), (0, 0), 0.8)
        flow = cv2.calcOpticalFlowFarneback(
            source,
            target,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=25,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )

        # Keep motion available just outside the coloured echo so the shader
        # can advect a storm edge into currently transparent pixels.
        valid = cv2.dilate(active.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        rgb = _pack_motion(flow, valid)

    if pad:
        rgb = rgb[pad:pad + previous.tile_size, pad:pad + previous.tile_size]

    output = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(output, format="PNG", compress_level=6)
    return output.getvalue()
