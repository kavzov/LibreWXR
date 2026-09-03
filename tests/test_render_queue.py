# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

import asyncio

import pytest

from librewxr.tiles.render_queue import BoundedRenderQueue


@pytest.mark.asyncio
async def test_queue_bounds_admitted_work_and_reports_waiters():
    queue = BoundedRenderQueue(worker_slots=2, queue_slots=1)
    release = asyncio.Event()

    async def hold_slot():
        async with queue:
            await release.wait()

    tasks = [asyncio.create_task(hold_slot()) for _ in range(5)]
    await asyncio.sleep(0)

    snapshot = queue.snapshot()
    assert snapshot["capacity"] == 3
    assert snapshot["inflight"] == 3
    assert snapshot["executor_queued"] == 1
    assert snapshot["waiting"] == 2
    assert snapshot["peak_waiting"] == 2

    release.set()
    await asyncio.gather(*tasks)
    assert queue.snapshot()["inflight"] == 0
    assert queue.snapshot()["waiting"] == 0
    assert queue.snapshot()["admitted_total"] == 5


@pytest.mark.asyncio
async def test_cancelled_waiter_is_removed_from_metrics():
    queue = BoundedRenderQueue(worker_slots=1, queue_slots=0)
    release = asyncio.Event()

    async def hold_slot():
        async with queue:
            await release.wait()

    holder = asyncio.create_task(hold_slot())
    await asyncio.sleep(0)
    waiter = asyncio.create_task(hold_slot())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert queue.snapshot()["waiting"] == 0
    release.set()
    await holder
