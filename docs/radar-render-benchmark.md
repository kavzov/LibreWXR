# Radar render benchmark

`scripts/benchmark_radar_tiles.py` is a deterministic benchmark for the
per-pixel radar hot path. It uses seed `20260902`, reports JSON, and separates
sampling, NWP blending, snow/colorization, blur, and PNG encoding.

Run both implementations after changing a kernel:

```sh
uv run python scripts/benchmark_radar_tiles.py --implementation python
uv run python scripts/benchmark_radar_tiles.py --implementation rust
```

The benchmark is intentionally independent of downloaded weather frames. The
production `/health` response complements it with observed `coordinates`,
`sampling`, `nwp_blend`, `snow`, `colorize`, `blur`, and `encode` averages from
real cold tile renders. Use both: the deterministic benchmark catches kernel
regressions, while production metrics reveal the workload mix.
