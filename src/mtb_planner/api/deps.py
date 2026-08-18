"""Shared request-time resources.

The route cache and the concurrency guard are process-wide: one cache and one
slot count per server, not one per request. They live here as FastAPI
dependencies rather than as module globals reached into directly, so a test can
substitute either through `app.dependency_overrides` instead of reassigning
another module's private name.
"""

from __future__ import annotations

from typing import Protocol

from mtb_planner.api.concurrency import PlanSlots
from mtb_planner.config.settings import get_settings
from mtb_planner.core.errors import BuildError, UnknownProfileError
from mtb_planner.cost.profiles import RideProfile, get_profile
from mtb_planner.cost.speed import Rider
from mtb_planner.services.graph_registry import LoadedGraph, get_registry
from mtb_planner.services.route_cache import RouteCache

_settings = get_settings()
_cache = RouteCache(maxsize=_settings.route_cache_size, ttl_s=_settings.route_cache_ttl_s)
_slots = PlanSlots(_settings.max_concurrent_plans)


def route_cache() -> RouteCache:
    """The process-wide route cache backing the GPX download links."""
    return _cache


def plan_slots() -> PlanSlots:
    """The process-wide cap on concurrent planning."""
    return _slots


def require_graph() -> LoadedGraph:
    """The loaded routing graph, or a 503 explaining why there is none."""
    return get_registry().require()


class ProfileOverrides(Protocol):
    """The controls every planning request carries.

    Structural rather than a base class, so point-to-point and loop requests
    share one resolver without one of them inheriting fields it does not use.
    """

    @property
    def profile(self) -> str: ...

    @property
    def tech_target(self) -> float | None: ...

    @property
    def detour(self) -> float | None: ...

    @property
    def power_w(self) -> float | None: ...


def resolve_profile(request: ProfileOverrides) -> RideProfile:
    """Look up the requested profile and apply the slider overrides.

    The loader raises `BuildError` for both an unknown id and a malformed
    profile file. Only the first is the caller's fault, so it is translated;
    the second stays a 500, which is what a broken deployment deserves.
    """
    try:
        profile = get_profile(request.profile)
    except BuildError as exc:
        raise UnknownProfileError(str(exc)) from exc
    return profile.with_overrides(tech_target=request.tech_target, detour=request.detour)


def resolve_rider(request: ProfileOverrides) -> Rider:
    """Build the rider the time model should use for this request."""
    return Rider() if request.power_w is None else Rider(power_w=request.power_w)
