"""The route cache that backs the GPX download link.

Three properties matter and nothing else does: a stored route comes back, a
route that has fallen out says so plainly rather than returning `None` for a
caller to misread, and the cache is safe to touch from the threadpool the
server answers plans on.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from mtb_planner.core.errors import RouteExpiredError
from mtb_planner.routing.result import Route, RouteStats
from mtb_planner.services.route_cache import CachedRoute, RouteCache

pytestmark = pytest.mark.unit


def a_route(distance_m: float = 12_340.0) -> Route:
    """A minimal route — the cache never looks inside beyond the stats."""
    return Route(
        arcs=(0, 1),
        nodes=(0, 1, 2),
        geometry=np.array([[50.100, 5.100], [50.101, 5.101], [50.102, 5.102]]),
        stats=RouteStats(distance_m=distance_m, n_edges=2, n_points=3, dismount_m=0.0),
    )


def an_entry(route_id: str = "abc123", distance_m: float = 12_340.0) -> CachedRoute:
    """A cache entry with the given id."""
    return CachedRoute(
        route_id=route_id,
        route=a_route(distance_m),
        profile_id="tech",
        profile_name="Technical",
    )


# --------------------------------------------------------------- storage


def test_a_stored_route_comes_back_unchanged():
    cache = RouteCache(maxsize=4, ttl_s=60.0)
    entry = an_entry()

    cache.put(entry)

    assert cache.get("abc123") is entry


def test_an_unknown_id_is_reported_as_expired_not_returned_as_nothing():
    # The endpoint turns this into a 410 with "plan it again"; returning None
    # would leave that decision to whichever caller forgot to check.
    cache = RouteCache(maxsize=4, ttl_s=60.0)

    with pytest.raises(RouteExpiredError, match="nope"):
        cache.get("nope")


def test_the_failure_message_tells_the_rider_what_to_do():
    cache = RouteCache(maxsize=4, ttl_s=60.0)

    with pytest.raises(RouteExpiredError, match="Plan it again"):
        cache.get("gone")


def test_storing_the_same_id_twice_keeps_the_newer_route():
    cache = RouteCache(maxsize=4, ttl_s=60.0)

    cache.put(an_entry("same", distance_m=1_000.0))
    cache.put(an_entry("same", distance_m=2_000.0))

    assert cache.get("same").route.stats.distance_m == 2_000.0
    assert len(cache) == 1


# --------------------------------------------------------------- bounds


def test_an_expired_route_is_gone():
    # A zero TTL expires every entry the moment it is stored, which tests the
    # expiry path without a sleep and without a fake clock.
    cache = RouteCache(maxsize=4, ttl_s=0.0)
    cache.put(an_entry())

    with pytest.raises(RouteExpiredError):
        cache.get("abc123")


def test_the_cache_stays_within_its_size_so_memory_is_bounded():
    cache = RouteCache(maxsize=2, ttl_s=60.0)

    for index in range(5):
        cache.put(an_entry(f"route-{index}"))

    assert len(cache) == 2


def test_the_most_recent_routes_are_the_ones_kept():
    cache = RouteCache(maxsize=2, ttl_s=60.0)

    cache.put(an_entry("old"))
    cache.put(an_entry("newer"))
    cache.put(an_entry("newest"))

    assert cache.get("newest")
    with pytest.raises(RouteExpiredError):
        cache.get("old")


def test_clearing_drops_everything():
    cache = RouteCache(maxsize=4, ttl_s=60.0)
    cache.put(an_entry())

    cache.clear()

    assert len(cache) == 0


# --------------------------------------------------------------- download name


def test_the_download_filename_says_what_the_ride_is():
    entry = an_entry(distance_m=42_400.0)

    assert entry.filename == "mtb-tech-42km.gpx"


def test_the_filename_carries_no_path_separators():
    # It goes straight into a Content-Disposition header, so anything
    # path-like in it would be a header-injection foothold.
    entry = an_entry(distance_m=1_000.0)

    assert "/" not in entry.filename
    assert "\\" not in entry.filename
    assert '"' not in entry.filename


# --------------------------------------------------------------- concurrency


def test_concurrent_writers_do_not_lose_routes():
    # `TTLCache` is not thread-safe and plans are answered on a threadpool, so
    # the cache locks. Without the lock this loses entries under contention.
    cache = RouteCache(maxsize=512, ttl_s=60.0)
    threads = [
        threading.Thread(target=cache.put, args=(an_entry(f"route-{index}"),))
        for index in range(64)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(cache) == 64
    assert all(cache.get(f"route-{index}") for index in range(64))
