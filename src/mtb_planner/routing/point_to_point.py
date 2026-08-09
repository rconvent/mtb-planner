"""Planning a route between two points.

Thin by design: materialise costs, snap, search, assemble. The same steps
serve the loop generator later, so the search is reached through one shared
entry point rather than each caller growing its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from mtb_planner.core.errors import NoRouteFoundError
from mtb_planner.core.geo import BBox, LatLon
from mtb_planner.cost.costvector import ArcCosts, materialise
from mtb_planner.cost.profiles import RideProfile
from mtb_planner.cost.speed import Rider
from mtb_planner.graph.spatial_index import SpatialIndex
from mtb_planner.graph.store import GraphStore
from mtb_planner.routing.budget import Deadline
from mtb_planner.routing.result import Route, build_route
from mtb_planner.routing.router import Router
from mtb_planner.routing.snap import Snap, snap_point


@dataclass(frozen=True)
class PointToPointPlan:
    """A planned route, with where each endpoint attached to the network."""

    route: Route
    start_snap: Snap
    end_snap: Snap
    settled: int


def plan_point_to_point(
    store: GraphStore,
    index: SpatialIndex,
    start: LatLon,
    end: LatLon,
    max_snap_distance_m: float,
    profile: RideProfile,
    rider: Rider | None = None,
    coverage: BBox | None = None,
    costs: ArcCosts | None = None,
    max_settled: int | None = None,
    deadline: Deadline | None = None,
) -> PointToPointPlan:
    """Plan the least-cost route between two coordinates.

    `costs` may be supplied to reuse a materialised cost vector across
    several requests on the same profile; otherwise it is built here.

    `deadline` bounds how long the search may run. An explicit `max_settled`
    takes precedence, since a caller asking for a specific ceiling means it.
    """
    arc_costs = costs or materialise(store, profile, rider)

    start_snap = snap_point(store, index, start, max_snap_distance_m, coverage)
    end_snap = snap_point(store, index, end, max_snap_distance_m, coverage)

    if start_snap.node == end_snap.node:
        raise NoRouteFoundError(
            reason="endpoints_coincide",
            detail="Start and finish snap to the same point on the network",
        )

    router = Router(store, arc_costs)
    if max_settled is not None:
        result = router.shortest(start_snap.node, end_snap.node, max_settled=max_settled)
    else:
        result = router.shortest_within(start_snap.node, end_snap.node, deadline=deadline)

    if result.truncated:
        # Deliberately a failure rather than a partial route. A path that stops
        # short of the destination is not a slightly worse route — it is a
        # route that leaves the rider somewhere they did not ask to be.
        raise NoRouteFoundError(
            reason="search_timeout",
            detail=(
                "Searching for a route between those points took longer than "
                "this server allows. They are probably very far apart."
            ),
        )

    if not result.reached:
        raise NoRouteFoundError(
            reason="disconnected",
            detail=(
                "No rideable connection between those points. They may sit in "
                "separate parts of the network, the area between them may be "
                "closed to bicycles, or it may be harder than this profile allows."
            ),
        )

    return PointToPointPlan(
        route=build_route(store, start_snap.node, result.arcs, arc_costs.time_s),
        start_snap=start_snap,
        end_snap=end_snap,
        settled=result.settled,
    )
