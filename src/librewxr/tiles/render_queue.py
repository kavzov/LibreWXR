# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Bound the work submitted to a render worker's compute executor.

``ThreadPoolExecutor`` uses an unbounded internal queue.  A large cold-tile
burst can therefore enqueue hundreds of expensive geometry jobs in every
process, even though only a handful can run.  This admission queue caps the
submitted set at ``worker_slots + queue_slots``; excess request coroutines
wait cheaply on the event loop until a submitted job completes.
"""

import asyncio


class BoundedRenderQueue:
    """Async admission control in front of one compute executor."""

    def __init__(self, worker_slots: int, queue_slots: int):
        if worker_slots < 1:
            raise ValueError("worker_slots must be positive")
        if queue_slots < 0:
            raise ValueError("queue_slots must not be negative")
        self.worker_slots = worker_slots
        self.queue_slots = queue_slots
        self.capacity = worker_slots + queue_slots
        self._semaphore = asyncio.Semaphore(self.capacity)
        self._inflight = 0
        self._waiting = 0
        self._peak_waiting = 0
        self._admitted_total = 0

    async def __aenter__(self) -> "BoundedRenderQueue":
        self._waiting += 1
        self._peak_waiting = max(self._peak_waiting, self._waiting)
        try:
            await self._semaphore.acquire()
        except BaseException:
            self._waiting -= 1
            raise
        self._waiting -= 1
        self._inflight += 1
        self._admitted_total += 1
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self._inflight -= 1
        self._semaphore.release()

    def snapshot(self) -> dict[str, int]:
        """Return a lock-free event-loop-local metrics snapshot."""
        return {
            "worker_slots": self.worker_slots,
            "queue_slots": self.queue_slots,
            "capacity": self.capacity,
            "inflight": self._inflight,
            "executor_queued": max(0, self._inflight - self.worker_slots),
            "waiting": self._waiting,
            "peak_waiting": self._peak_waiting,
            "admitted_total": self._admitted_total,
        }
