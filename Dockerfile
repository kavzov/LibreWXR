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
# plus the abi3 Rust sampling kernels. The container defaults to strict native
# mode below; an operator can still explicitly force the Python path with
# LIBREWXR_NATIVE_RENDER=off for diagnosis.
RUN pip install --no-cache-dir '.[mcp]' /wheels/*.whl \
    && rm -rf /wheels

# Container builds always carry the wheel, so fail closed if it ever becomes
# unloadable instead of silently falling back to the slower NumPy path.
ENV LIBREWXR_NATIVE_RENDER=on

EXPOSE 8080

CMD ["python", "-m", "librewxr.main"]
