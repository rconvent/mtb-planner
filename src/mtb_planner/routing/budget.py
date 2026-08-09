"""The wall-clock budget for one planning request.

A request must not run forever, and the ceiling has to be expressed in seconds
because that is the only unit a rider or an HTTP client cares about. The
kernels cannot help: Numba's nopython mode has no access to `time`, so a
compiled search cannot check a clock and stop itself.

What a kernel *can* take is a ceiling on how many nodes it settles. So the
clock is enforced from outside, by running the search under a settle ceiling
and raising that ceiling between rounds for as long as the budget allows. The
ceiling grows geometrically, which bounds the wasted work: with a growth factor
of eight, every round before the last one together costs about a seventh of the
last, so the common case — a search that finishes inside the first ceiling —
pays nothing at all.

The alternative, converting seconds into a settle count with a measured
throughput constant, was rejected: the constant is machine-specific, silently
wrong on hardware it was not measured on, and wrong in the dangerous direction
on a slower machine.

`time.monotonic` throughout, so a clock adjustment mid-request cannot make a
deadline jump.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

#: Settle ceiling for the first round. Comfortably covers a whole
#: country-sized graph — Luxembourg is 164,650 nodes — so a typical search
#: finishes in one round and never pays for the deepening machinery.
INITIAL_SETTLE_CEILING = 250_000

#: How much the ceiling grows per round. Eight keeps the total wasted work
#: under a seventh of the final round while reaching a 4M-node graph in three.
SETTLE_GROWTH = 8


@dataclass(frozen=True)
class Deadline:
    """A moment on the monotonic clock after which work should stop.

    Phase 8's loop generator consults this between candidates so it can return
    its best-so-far rather than failing. A single point-to-point search has no
    best-so-far — a path that stops short of the destination is not a worse
    route, it is the wrong answer — so there, expiry is reported as a failure.
    """

    expires_at: float

    @classmethod
    def after(cls, seconds: float) -> Deadline:
        """A deadline `seconds` from now."""
        return cls(expires_at=time.monotonic() + seconds)

    @property
    def remaining_s(self) -> float:
        """Seconds left, never negative."""
        return max(0.0, self.expires_at - time.monotonic())

    @property
    def expired(self) -> bool:
        """Whether the budget is spent."""
        return self.remaining_s <= 0.0
