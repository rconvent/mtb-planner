"""Shared request-time resources.

The route cache and the concurrency guard are process-wide: one cache and one
slot count per server, not one per request. They live here as FastAPI
dependencies rather than as module globals reached into directly, so a test can
substitute either through `app.dependency_overrides` instead of reassigning
another module's private name.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from planner_core.api.concurrency import PlanSlots
from planner_core.config.settings import get_settings
from planner_core.core.errors import BuildError, UnknownProfileError
from planner_core.cost.profiles import RideProfile, get_profile
from planner_core.services.graph_registry import LoadedGraph, get_registry
from planner_core.services.route_cache import RouteCache
from planner_core.sport import Athlete, active_sport


@lru_cache(maxsize=1)
def route_cache() -> RouteCache:
    """The process-wide route cache backing the GPX download links.

    Built on first use rather than at import, because its size comes from the
    active sport's settings and importing this module must not require a sport.
    """
    settings = get_settings()
    return RouteCache(maxsize=settings.route_cache_size, ttl_s=settings.route_cache_ttl_s)


@lru_cache(maxsize=1)
def plan_slots() -> PlanSlots:
    """The process-wide cap on concurrent planning."""
    return PlanSlots(get_settings().max_concurrent_plans)


def require_graph() -> LoadedGraph:
    """The loaded routing graph, or a 503 explaining why there is none."""
    return get_registry().require()


class ProfileOverrides(Protocol):
    """The controls every planning request carries.

    Structural rather than a base class, so point-to-point and loop requests
    share one resolver without one of them inheriting fields it does not use.
    """

    @property
    def profile(self) -> str | None: ...

    @property
    def tech_target(self) -> float | None: ...

    @property
    def detour(self) -> float | None: ...

    @property
    def effort(self) -> float | None: ...


def resolve_profile(request: ProfileOverrides) -> RideProfile:
    """Look up the requested profile and apply the slider overrides.

    The loader raises `BuildError` for both an unknown id and a malformed
    profile file. Only the first is the caller's fault, so it is translated;
    the second stays a 500, which is what a broken deployment deserves.
    """
    try:
        profile = get_profile(request.profile or active_sport().default_profile)
    except BuildError as exc:
        raise UnknownProfileError(str(exc)) from exc
    return profile.with_overrides(tech_target=request.tech_target, detour=request.detour)


def resolve_athlete(request: ProfileOverrides) -> Athlete:
    """Build the athlete the time model should use for this request."""
    return active_sport().athlete(request.effort)
