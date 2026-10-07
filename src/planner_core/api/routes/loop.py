"""Generating loops from a point.

A plain `def`, like every other planning endpoint, so FastAPI runs it on the
threadpool rather than blocking the event loop with a dozen compiled searches.

Loops go into the same route cache as point-to-point routes, so
`/api/routes/{id}/gpx` serves them without knowing which endpoint produced them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from planner_core.api.concurrency import PlanSlots
from planner_core.api.deps import (
    plan_slots,
    require_graph,
    resolve_athlete,
    resolve_profile,
    route_cache,
)
from planner_core.api.schemas.common import ApiResponse, LatLonModel
from planner_core.api.schemas.planning import (
    LoopModel,
    LoopRequest,
    LoopResponse,
    SnapModel,
    character_of,
    elevation_of,
    geometry_of,
    segments_of,
    stats_of,
)
from planner_core.config.settings import get_settings
from planner_core.graph.store import GraphStore
from planner_core.routing import loop as engine
from planner_core.routing.budget import Deadline
from planner_core.routing.snap import Snap
from planner_core.services.graph_registry import LoadedGraph
from planner_core.services.route_cache import CachedRoute, RouteCache, new_route_id

router = APIRouter(tags=["planning"])

#: Below this, overlap is not worth mentioning. Warnings are rendered to the
#: nearest per cent, and "0% of this loop is ridden twice" tells a rider nothing.
_OVERLAP_WORTH_MENTIONING = 0.01


def _warnings(candidate: engine.LoopCandidate, snap: Snap, max_snap_m: float) -> tuple[str, ...]:
    """Anything about this loop the rider should know before setting off."""
    notes: list[str] = []
    if snap.distance_m > max_snap_m / 2:
        notes.append(f"The start was moved {snap.distance_m:.0f} m to reach a trail.")
    if not candidate.within_budget:
        actual = candidate.metrics.time_s / 60.0
        notes.append(f"This loop takes about {actual:.0f} minutes, outside the time asked for.")
    if candidate.route.stats.walk_m > 0:
        notes.append(f"{candidate.route.stats.walk_m:.0f} m of this loop must be walked.")
    if candidate.metrics.overlap_ratio >= _OVERLAP_WORTH_MENTIONING:
        notes.append(f"{candidate.metrics.overlap_ratio:.0%} of this loop is ridden twice.")
    return tuple(notes)


def _as_model(
    store: GraphStore,
    candidate: engine.LoopCandidate,
    route_id: str,
    snap: Snap,
    max_snap_m: float,
) -> LoopModel:
    """Shape a generated loop for the wire."""
    return LoopModel(
        route_id=route_id,
        geometry=geometry_of(candidate.route),
        stats=stats_of(candidate.route),
        character=character_of(candidate.metrics),
        segments=segments_of(store, candidate.route),
        elevation=elevation_of(store, candidate.route),
        start_snap=SnapModel(
            point=LatLonModel.from_domain(snap.point),
            distance_m=round(snap.distance_m, 1),
        ),
        gpx_url=f"/api/routes/{route_id}/gpx",
        overlap_ratio=round(candidate.metrics.overlap_ratio, 3),
        score=round(candidate.score, 4),
        within_budget=candidate.within_budget,
        warnings=_warnings(candidate, snap, max_snap_m),
    )


@router.post("/api/routes/loop", response_model=ApiResponse[LoopResponse])
def plan_loop(
    request: LoopRequest,
    graph: LoadedGraph = Depends(require_graph),
    cache: RouteCache = Depends(route_cache),
    slots: PlanSlots = Depends(plan_slots),
) -> ApiResponse[LoopResponse]:
    """Generate loops from a point that fit a time budget."""
    settings = get_settings()
    profile = resolve_profile(request)
    athlete = resolve_athlete(request)
    deadline = Deadline.after(settings.route_timeout_s)

    query = engine.LoopRequest(
        start=request.start.to_domain(),
        budget_s=request.budget_s,
        profile=profile,
        athlete=athlete,
        variants=request.variants,
        tolerance=settings.budget_tolerance,
        max_overlap=settings.max_overlap_ratio,
        max_snap_distance_m=settings.max_snap_distance_m,
    )

    with slots.hold():
        plan = engine.plan_loops(
            graph.store,
            graph.index,
            query,
            coverage=graph.bbox,
            deadline=deadline,
        )

    loops: list[LoopModel] = []
    for candidate in plan.candidates:
        route_id = new_route_id()
        cache.put(
            CachedRoute(
                route_id=route_id,
                route=candidate.route,
                profile_id=profile.id,
                profile_name=profile.name,
                metrics=candidate.metrics,
                score=round(candidate.score, 4),
                within_budget=candidate.within_budget,
            )
        )
        loops.append(
            _as_model(
                graph.store,
                candidate,
                route_id,
                plan.start_snap,
                settings.max_snap_distance_m,
            )
        )

    return ApiResponse.ok(
        LoopResponse(loops=tuple(loops)),
        profile=profile.id,
        artifact=graph.artifact,
        settled=plan.settled,
        truncated=plan.truncated,
        seed=request.seed,
    )
