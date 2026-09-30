# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Regression tests for HTTP retries and exception propagation."""

import httpx
import pytest

from librewxr.data.retry import retry_get


@pytest.mark.parametrize("error_type", [httpx.DecodingError, httpx.ConnectError])
@pytest.mark.parametrize("recover", [False, True])
async def test_retry_get_transient_errors(error_type, recover):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if recover and calls == 3:
            return httpx.Response(200, content=b"recovered")
        raise error_type("transient failure", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await retry_get(
            client, "https://example.test/data", retries=2, delay=0.0,
        )

    assert calls == 3
    if recover:
        assert response.status_code == 200
        assert response.content == b"recovered"
    else:
        assert response is None


async def test_retry_get_does_not_mask_or_retry_status_error():
    request = httpx.Request("GET", "https://example.test/data")
    error = httpx.HTTPStatusError(
        "server error", request=request, response=httpx.Response(500, request=request),
    )
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        raise error

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await retry_get(client, str(request.url), retries=2, delay=0.0)

    assert caught.value is error
    assert calls == 1
