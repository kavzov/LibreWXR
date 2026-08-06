# Weather tile rendering benchmark

This benchmark covers the scalar weather rendering path independently of
`pytest-benchmark`. Run it from the repository root:

```bash
.venv/bin/python scripts/benchmark_weather_tiles.py \
  --iterations 3 --parallel 4 --native-render off \
  --output /tmp/weather-tiles.json
```

The deterministic fixture uses two native valid times on a compact regular
global grid. Every run covers temperature, pressure, relative humidity, and
wind speed at z3/z6/z10, 256/512 px, and PNG/WebP. It measures cold and warm
sampling plans, cold and warm encoded-tile cache access, concurrent p50/p95,
encoded bytes, and peak Python/NumPy temporary allocations through
`tracemalloc`.

## Optimization stages

Measurements below were taken on the same machine with Python 3.13.3, three
warm iterations and four concurrent requests. Values are means across fields,
formats, and zooms for each tile size. Absolute timings vary by host; the
relative comparison is the intended signal.

| Stage | Warm 256 | Warm 512 | Peak 256 | Peak 512 | Parallel cold p95 256 | Parallel cold p95 512 |
|---|---:|---:|---:|---:|---:|---:|
| Before | 16.03 ms | 59.10 ms | 7.19 MiB | 28.76 MiB | 23.09 ms | 91.45 ms |
| Quantized PNG encoder | 16.14 ms | 58.44 ms | 7.19 MiB | 28.76 MiB | 22.18 ms | 88.40 ms |
| Global sampling fast path | 11.40 ms | 39.45 ms | 2.84 MiB | 11.35 MiB | 14.63 ms | 55.45 ms |
| Reused color/derived buffers | 10.72 ms | 34.19 ms | 2.42 MiB | 9.66 MiB | 14.53 ms | 50.51 ms |

The encoder stage is intentionally shown separately: it reduced typical PNG
size without making warm rendering slower. Sampling then removed duplicate
tile-coordinate, feather, decoded-value, weight, and validity rasters when IFS
is the sole participant. The final stage kept palette output pixel-identical
and moved derived calculations from float64 to float32; the maximum humidity
difference against the previous formula is below 0.00024 percentage points.

## Before/after detail

| Metric | 256 before | 256 after | 512 before | 512 after |
|---|---:|---:|---:|---:|
| Cold sampling-plan render | 19.98 ms | 14.78 ms | 65.21 ms | 41.85 ms |
| Warm sampling-plan render | 16.03 ms | 10.72 ms | 59.10 ms | 34.19 ms |
| Cold encoded-tile cache | 14.34 ms | 9.76 ms | 57.22 ms | 32.74 ms |
| Warm tile-cache p50 | 0.0003 ms | 0.0004 ms | 0.0003 ms | 0.0004 ms |
| Warm tile-cache p95 | 0.0011 ms | 0.0011 ms | 0.0012 ms | 0.0012 ms |
| Parallel cold p50 | 22.57 ms | 13.93 ms | 89.57 ms | 48.05 ms |
| Parallel cold p95 | 23.09 ms | 14.53 ms | 91.45 ms | 50.51 ms |
| Parallel warm p50 | 0.0004 ms | 0.0004 ms | 0.0004 ms | 0.0004 ms |
| Parallel warm p95 | 0.0015 ms | 0.0015 ms | 0.0016 ms | 0.0015 ms |
| Cold peak temporary memory | 9.63 MiB | 7.32 MiB | 38.51 MiB | 29.26 MiB |
| Warm peak temporary memory | 7.19 MiB | 2.42 MiB | 28.76 MiB | 9.66 MiB |

Average encoded payload sizes:

| Format | 256 before | 256 after | Change | 512 before | 512 after | Change |
|---|---:|---:|---:|---:|---:|---:|
| PNG | 4,969 B | 3,548 B | −28.6% | 11,794 B | 7,933 B | −32.7% |
| Lossless WebP | 3,609 B | 3,605 B | −0.1% | 8,177 B | 8,138 B | −0.5% |

The PNG visual-regression fixture requires exact alpha preservation, mean RGB
error below 3 levels, p95 at most 10 levels, and maximum error at most 20
levels. The lossless mode and all exact PNG8 tiles remain pixel-exact.

## Optional native sampling result

The native experiment was gated on a CPU profile of the optimized NumPy run
(`--iterations 1 --parallel 1 --native-render off`). Excluding the benchmark's
explicit garbage collections, `_numpy_sample` was the largest application
compute site: 1.34 seconds of self time across 576 calls (1.75 seconds
cumulative). The next application operation, palette colourization, used
0.25 seconds; WebP encoding spent 1.62 seconds in its external encoder. This
confirmed sampling/interpolation as a material CPU bottleneck before any Rust
code was selected for the production path.

The PyO3 candidate was measured only after the optimized NumPy baseline above.
Both runs used the same renderer and deterministic fixtures; only
`LIBREWXR_NATIVE_RENDER=off|on` changed. Each row averages all fields, formats,
and zooms for that size. Two iterations were used for the concurrency sweep.

| Concurrency | Warm 256 NumPy → Rust | Warm 512 NumPy → Rust | Cold p95 256 NumPy → Rust | Cold p95 512 NumPy → Rust |
|---:|---:|---:|---:|---:|
| 1 | 10.80 → 8.08 ms (−25.2%) | 35.92 → 27.54 ms (−23.3%) | 9.70 → 7.55 ms (−22.1%) | 34.47 → 26.90 ms (−22.0%) |
| 8 | 10.85 → 8.19 ms (−24.5%) | 35.93 → 27.19 ms (−24.3%) | 31.02 → 20.03 ms (−35.4%) | 93.16 → 66.84 ms (−28.3%) |
| 32 | 11.82 → 8.79 ms (−25.6%) | 39.75 → 28.50 ms (−28.3%) | 114.64 → 72.71 ms (−36.6%) | 339.67 → 222.10 ms (−34.6%) |

Warm traced temporary memory fell from 2.42 to 1.74 MiB for 256 px and
from 9.66 to 6.79 MiB for 512 px. Total sweep time fell by 15.3%, 19.2%, and
25.7% at concurrency 1, 8, and 32 respectively. These measurements include
Python argument validation, the Python/Rust boundary, output allocation, LUT
colourization, and PNG/WebP encoding.

The result justified retaining the optional extension: boundary cost does not
erase the gain, and scaling improves when concurrent requests can run native
kernels outside the GIL. No Rayon dependency or internal thread pool is used.
Parity tests permit at most half one field encoding step and cover nearest,
bilinear, fused temporal interpolation, dateline wrapping, grid boundaries,
nodata, humidity, wind speed, and deterministic random inputs.
