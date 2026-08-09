"""Shared route invariants.

Exact-path assertions against a heuristic search are the textbook definition of
a brittle test: any tuning of the cost model rewrites them wholesale. These
invariants hold for *any* correct router, so they survive that tuning and
still catch real breakage.
"""

from __future__ import annotations

import pytest

from mtb_planner.graph.store import GraphStore
from mtb_planner.ingest.way_filter import BikeAccess
from mtb_planner.routing.loop import LoopCandidate, LoopRequest
from mtb_planner.routing.result import Route


def _assert_arcs_are_connected(store: GraphStore, route: Route) -> None:
    """Each arc must start where the previous one ended."""
    for position, arc in enumerate(route.arcs):
        tail = store.arc_tail(arc)
        head = int(store.arc_head[arc])
        assert tail == route.nodes[position], (
            f"arc {arc} at position {position} starts at node {tail}, "
            f"but the route is at {route.nodes[position]}"
        )
        assert head == route.nodes[position + 1]


def _assert_arcs_are_traversable(store: GraphStore, route: Route) -> None:
    """No route may use an arc the direction rules forbid."""
    for arc in route.arcs:
        assert store.arc_allowed[arc], f"arc {arc} is not traversable"


def _assert_no_forbidden_ways(store: GraphStore, route: Route) -> None:
    """A legally closed way must never appear. This is not a soft preference."""
    for arc in route.arcs:
        edge = int(store.arc_edge[arc])
        access = int(store.edge_access[edge])
        assert access != BikeAccess.NO, (
            f"route uses edge {edge} (OSM way {int(store.edge_way_id[edge])}), "
            "which bicycles are not permitted on"
        )


def _assert_stats_match_geometry(store: GraphStore, route: Route) -> None:
    """Reported distance must be the sum of the edges actually ridden."""
    expected = sum(float(store.edge_length_m[store.arc_edge[arc]]) for arc in route.arcs)
    assert route.stats.distance_m == pytest.approx(expected, rel=1e-9)
    assert route.stats.n_edges == len(route.arcs)
    assert route.stats.n_points == route.geometry.shape[0]
    assert route.geometry.shape[1] == 2


def assert_valid_route(
    store: GraphStore,
    route: Route,
    start_node: int | None = None,
    end_node: int | None = None,
) -> None:
    """Assert every property a correct route must have."""
    assert route.arcs, "route has no arcs"
    assert len(route.nodes) == len(route.arcs) + 1

    if start_node is not None:
        assert route.nodes[0] == start_node
    if end_node is not None:
        assert route.nodes[-1] == end_node

    _assert_arcs_are_connected(store, route)
    _assert_arcs_are_traversable(store, route)
    _assert_no_forbidden_ways(store, route)
    _assert_stats_match_geometry(store, route)


def assert_valid_loop(
    store: GraphStore,
    candidate: LoopCandidate,
    start_node: int,
    request: LoopRequest,
) -> None:
    """Assert every property a correct loop must have.

    Everything `assert_valid_route` demands, plus the three things that make a
    loop a loop: it comes back, it does not merely retrace itself, and the
    figures reported for it are the ones measured from the arcs ridden.
    """
    route = candidate.route
    assert_valid_route(store, route, start_node=start_node, end_node=start_node)
    assert route.is_loop, "loop does not return to its start"

    assert candidate.metrics.overlap_ratio <= request.max_overlap, (
        f"loop repeats {candidate.metrics.overlap_ratio:.0%} of its length, "
        f"above the {request.max_overlap:.0%} limit"
    )

    # Measured, never inferred from the cost that drove the search.
    assert candidate.metrics.distance_m == pytest.approx(route.stats.distance_m, rel=1e-6)
    assert candidate.metrics.time_s == pytest.approx(route.stats.moving_time_s, rel=1e-6)

    if candidate.within_budget:
        assert abs(candidate.metrics.budget_error(request.budget_s)) <= request.tolerance
