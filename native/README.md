# Optional native weather sampling

LibreWXR's base Hatchling package does not build or install this crate. The
application always includes a NumPy implementation and therefore installs and
runs normally without Rust, Cargo, maturin, or a native wheel.

Build an ABI3 wheel with Python 3.11+ compatibility:

```bash
maturin build --release --manifest-path native/Cargo.toml \
  --interpreter .venv/bin/python
.venv/bin/pip install native/target/wheels/librewxr_native-*.whl
```

Production images should build the wheel in a separate builder stage and copy
only the wheel into the runtime image. The extension has no Rayon dependency,
creates no thread pool, and releases the GIL around each sampling or derived
field kernel. Existing Uvicorn worker and request-executor limits remain the
only concurrency controls.

Use `LIBREWXR_NATIVE_RENDER=auto` (default) for automatic fallback, `on` to
require the wheel at startup, or `off` to force NumPy.
