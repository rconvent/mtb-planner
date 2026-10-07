"""Capping how much planning work runs at once.

Planning is CPU-bound NumPy and Numba on a threadpool. Left uncapped, three
impatient clicks each take a core and the machine spends its time context
switching instead of routing. A semaphore turns that into a queue.

Refusing outright rather than queueing indefinitely is deliberate: a rider
staring at a spinner cannot tell "busy" from "broken", and a 503 with
`Retry-After` lets the client say which it is.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from planner_core.core.errors import ServerBusyError


class PlanSlots:
    """A fixed number of concurrent planning slots."""

    def __init__(self, limit: int) -> None:
        self._semaphore = threading.Semaphore(limit)

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Occupy a slot for the duration of the block, or refuse.

        Non-blocking: the caller is told to come back rather than left waiting
        behind work it cannot see.
        """
        if not self._semaphore.acquire(blocking=False):
            msg = "Too many routes are being planned at once. Try again shortly."
            raise ServerBusyError(msg)
        try:
            yield
        finally:
            self._semaphore.release()
