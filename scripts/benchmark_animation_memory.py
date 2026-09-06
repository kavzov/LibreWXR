#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Deterministic animation benchmark; run each revision in a fresh process.

Compare output_sha256 as well as seconds and peak_rss_mib. No network, cache
deletion or production data is needed. Use identical Python/OpenCV versions.
"""
import argparse
import hashlib
import json
import resource
import sys
import time

import cv2
import numpy as np

from librewxr.data.nowcast import NowcastFrame, NowcastGenerator
from librewxr.data.store import RadarFrame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1536)
    parser.add_argument("--regions", type=int, default=6)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--substeps", type=int, default=2)
    args = parser.parse_args()
    if min(args.size, args.regions, args.steps) < 1 or args.substeps < 2:
        parser.error("positive dimensions and substeps >= 2 required")
    cv2.setNumThreads(2)
    rng = np.random.default_rng(20260906)
    latest, previous, flows = {}, {}, {}
    for index in range(args.regions):
        name = f"R{index}"
        # Distinct shapes also exercise the interpolation grid cache.
        shape = (args.size + index * 8, args.size + index * 16)
        latest[name] = rng.integers(0, 256, shape, dtype=np.uint8)
        previous[name] = np.roll(latest[name], -2, axis=1)
        flows[name] = rng.uniform(-3, 3, (64, 64, 2)).astype(np.float32)
    observed = [RadarFrame(timestamp=0, regions=previous),
                RadarFrame(timestamp=300, regions=latest)]
    forecast = [NowcastFrame(timestamp=300 + step * 300, regions=latest,
                            blend_weight=1 - step / (args.steps + 1))
                for step in range(1, args.steps + 1)]
    started = time.perf_counter()
    frames, valid = NowcastGenerator._generate_animation_sync(
        observed, forecast, flows, 300, args.substeps,
    )
    elapsed = time.perf_counter() - started
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(repr((frame.timestamp, frame.blend_weight, frame.period)).encode())
        for name, data in sorted(frame.regions.items()):
            digest.update(name.encode())
            digest.update(data.tobytes())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({
        "fixture": vars(args), "seconds": elapsed,
        "peak_rss_mib": rss / (1024 ** 2 if sys.platform == "darwin" else 1024),
        "frames": len(frames), "valid": sorted(valid),
        "output_sha256": digest.hexdigest(), "opencv": cv2.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
