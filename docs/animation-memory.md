# Bounded animation working memory

The display-only animation generator processes one region across the output
timeline before moving to the next. It reuses that region's full-resolution
flow, then releases it. It does not retain full-resolution flows for all
regions or build coordinate grids ignored by OpenCV's relative-map warp.
Observed interpolation and fallback interpolation retain the same algorithm,
forecast weights, timestamps, resolution and number of frames. Publication
still happens atomically through the existing NowcastStore paths.

The shared NWP interpolation helper caches read-only broadcast views backed by
two float32 coordinate axes, not two dense grids per shape. Coordinate values
and remap arithmetic are unchanged. Forward remap buffers are released before
the backward maps are allocated.

## Verification

Run each revision in a fresh process, with the same Python, NumPy and OpenCV:

```sh
python scripts/benchmark_animation_memory.py --size 1536 --regions 6 --steps 12
pytest tests/test_nowcast.py tests/test_ecmwf_interpolation.py
```

Compare `output_sha256` before comparing time/RSS. The synthetic benchmark is
not a tile-throughput test: production rollout must also compare foreground
RPS, tile/frame p95/p99 and update-cycle duration over several fetch cycles.
Keep renderer images, CPU/RAM/thread limits and cache budgets unchanged.
Use the prior pipeline image for rollback; never remove weather snapshots.

Initial Linux/OpenCV 5.0 comparison (6 regions, 1536px, 12 steps) reduced
peak process RSS from 659 to 339 MiB with the same frame digest; calculation
time was 1.17 versus 0.64 seconds. These are microbenchmark observations,
not a promise of the same percentage improvement for whole-server RAM/RPS.
