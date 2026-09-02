FROM python:3.12-slim AS native-builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends cargo gcc \
    && rm -rf /var/lib/apt/lists/*

COPY native/ /build/native/
RUN pip wheel --no-cache-dir --no-deps /build/native -w /wheels


FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY --from=native-builder /wheels/ /wheels/

# Install with the [mcp] extra so the MCP HTTP transport mounts on startup,
# plus the optional abi3 Rust sampling kernels. ``auto`` selects them at
# runtime and still permits an operator to force the Python path with
# LIBREWXR_NATIVE_RENDER=off.
RUN pip install --no-cache-dir '.[mcp]' /wheels/*.whl \
    && rm -rf /wheels

EXPOSE 8080

CMD ["python", "-m", "librewxr.main"]
