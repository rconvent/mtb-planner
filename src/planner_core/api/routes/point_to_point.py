"""Planning a route between two points.

The path operation is a plain `def`, not `async def`, so FastAPI runs it on the
threadpool. Routing is CPU-bound NumPy and Numba; doing it inside `async def`
would block the event loop and freeze every other request for the duration.
The kernels release the GIL, so threads genuinely overlap.
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
    PlanRequest,
    PlanResponse,
    RouteModel,
    SnapModel,
    elevation_of,
    geometry_of,
    segments_of,
    stats_of,
)
from planner_core.config.settings import get_settings
from planner_core.graph.store import GraphStore
from planner_core.routing.budget import Deadline
from planner_core.routing.point_to_point import PointToPointPlan, plan_point_to_point
from planner_core.services.graph_registry import LoadedGraph
from planner_core.services.route_cache import CachedRoute, RouteCache, new_route_id

router = APIRouter(tags=["planning"])


def _as_model(
    store: GraphStore,
    plan: PointToPointPlan,
    route_id: str,
    warnings: tuple[str, ...],
) -> RouteModel:
    """Shape a planned route for the wire."""
    return RouteModel(
        route_id=route_id,
        within_budget=plan.within_budget,
        geometry=geometry_of(plan.route),
        stats=stats_of(plan.route),
        segments=segments_of(store, plan.route),
        elevation=elevation_of(store, plan.route),
        start_snap=SnapModel(
            point=LatLonModel.from_domain(plan.start_snap.point),
            distance_m=round(plan.start_snap.distance_m, 1),
        ),
        end_snap=SnapModel(
            point=LatLonModel.from_domain(plan.end_snap.point),
            distance_m=round(plan.end_snap.distance_m, 1),
        ),
        gpx_url=f"/api/routes/{route_id}/gpx",
        warnings=warnings,
    )


def _warnings(
    plan: PointToPointPlan, max_snap_distance_m: float, budget_minutes: int | None
) -> tuple[str, ...]:
    """Anything about this route the rider should know before setting off."""
    notes: list[str] = []
    for label, snap in (("start", plan.start_snap), ("finish", plan.end_snap)):
        if snap.distance_m > max_snap_distance_m / 2:
            notes.append(f"The {label} was moved {snap.distance_m:.0f} m to reach a trail.")
    if plan.route.stats.walk_m > 0:
        notes.append(f"{plan.route.stats.walk_m:.0f} m of this route must be walked.")

    # A budget that cannot be met is worth saying out loud. The rider gets the
    # route they asked for either way — the destination is not negotiable — but
    # being told it runs long beats discovering it at dusk.
    if budget_minutes is not None and plan.within_budget is False:
        notes.append(
            f"The best way there takes about {plan.route.stats.moving_time_min:.0f} min, "
            f"longer than the {budget_minutes} you allowed."
        )
    return tuple(notes)


@router.post("/api/routes/point-to-point", response_model=ApiResponse[PlanResponse])
def plan_route(
    request: PlanRequest,
    graph: LoadedGraph = Depends(require_graph),
    cache: RouteCache = Depends(route_cache),
    slots: PlanSlots = Depends(plan_slots),
) -> ApiResponse[PlanResponse]:
    """Plan a route between two points."""
    settings = get_settings()
    profile = resolve_profile(request)
    athlete = resolve_athlete(request)

    # Started before the cost vector is materialised, so the ceiling covers
    # everything this request does rather than the search alone.
    deadline = Deadline.after(settings.route_timeout_s)

    # The slot is held across the search only. Shaping the response is cheap
    # and holding a slot through it would cut throughput for nothing.
    with slots.hold():
        plan = plan_point_to_point(
            graph.store,
            graph.index,
            request.start.to_domain(),
            request.end.to_domain(),
            max_snap_distance_m=settings.max_snap_distance_m,
            profile=profile,
            athlete=athlete,
            coverage=graph.bbox,
            deadline=deadline,
            budget_s=request.budget_s,
        )

    route_id = new_route_id()
    cache.put(
        CachedRoute(
            route_id=route_id,
            route=plan.route,
            profile_id=profile.id,
            profile_name=profile.name,
        )
    )

    return ApiResponse.ok(
        PlanResponse(
            routes=(
                _as_model(
                    graph.store,
                    plan,
                    route_id,
                    _warnings(plan, settings.max_snap_distance_m, request.budget_minutes),
                ),
            )
        ),
        profile=profile.id,
        artifact=graph.artifact,
        settled=plan.settled,
        detour=round(plan.detour, 2),
    )
