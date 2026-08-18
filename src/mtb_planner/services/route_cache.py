"""Holding planned routes long enough to download them.

A browser cannot trigger a file download from a POST body without blob
gymnastics, so a planned route gets an id and a URL, and the GPX is fetched
with a plain link. That means the route has to outlive the request that made
it — but only briefly.

Deliberately in memory and bounded. Routes are cheap to regenerate, which
makes persistence a liability rather than a feature.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from cachetools import TTLCache

from mtb_planner.core.errors import RouteExpiredError
from mtb_planner.routing.candidates import LoopMetrics
from mtb_planner.routing.result import Route

#: How many hex characters a route id carries. Twelve is 48 bits — ample
#: against collision within a cache of a few hundred entries, and short enough
#: to paste into a bug report.
ROUTE_ID_LENGTH = 12

#: What a route id must look like. The retrieval endpoints validate against
#: this so a mistyped path is a bad request rather than a route reported as
#: expired — `/api/routes/loop` must not answer "plan it again".
ROUTE_ID_PATTERN = rf"^[0-9a-f]{{{ROUTE_ID_LENGTH}}}$"


def new_route_id() -> str:
    """Mint an id for a freshly planned route.

    Kept beside `ROUTE_ID_PATTERN` so the generator and the validator cannot
    drift apart.
    """
    return uuid.uuid4().hex[:ROUTE_ID_LENGTH]


@dataclass(frozen=True)
class CachedRoute:
    """A planned route, with enough context to name its file and re-read it.

    The judgement fields are optional because a point-to-point route has none:
    it is the only route between two points that the profile prefers, so there
    is nothing to rank it against. A loop has all three, and without them a
    shared link came back stripped of the character, overlap and budget verdict
    that were shown when it was planned.
    """

    route_id: str
    route: Route
    profile_id: str
    profile_name: str
    metrics: LoopMetrics | None = None
    score: float | None = None
    within_budget: bool | None = None

    @property
    def filename(self) -> str:
        """A download name that says what the ride is."""
        return f"mtb-{self.profile_id}-{self.route.stats.distance_km:.0f}km.gpx"


class RouteCache:
    """Recently planned routes, keyed by id.

    `cachetools.TTLCache` is not thread-safe and the server answers plans on a
    thread pool, so every access takes a lock.
    """

    def __init__(self, maxsize: int, ttl_s: float) -> None:
        self._lock = threading.Lock()
        self._entries: TTLCache[str, CachedRoute] = TTLCache(maxsize=maxsize, ttl=ttl_s)

    def put(self, entry: CachedRoute) -> None:
        """Store a planned route."""
        with self._lock:
            self._entries[entry.route_id] = entry

    def get(self, route_id: str) -> CachedRoute:
        """Retrieve a route, or say plainly that it has gone."""
        with self._lock:
            entry = self._entries.get(route_id)
        if entry is None:
            msg = f"Route {route_id} is no longer available. Plan it again."
            raise RouteExpiredError(msg)
        return entry

    def __len__(self) -> int:
        """How many routes are currently held."""
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Drop everything held."""
        with self._lock:
            self._entries.clear()
