"""Snapping and search.

Route assertions go through the shared invariants in `tests/assertions.py`
rather than pinning exact paths: a heuristic search may legitimately return
either of two equal-cost routes, and a test that forbids that is testing the
tie-break, not the router.
"""

from __future__ import annotations

import numpy as np
import pytest

from planner_core.core.errors import (
    NoRouteFoundError,
    PointOffNetworkError,
    PointOutsideCoverageError,
)
from planner_core.core.geo import LatLon
from planner_core.graph.spatial_index import SpatialIndex
from planner_core.graph.store import GraphStore
from planner_core.routing.astar import SearchResult, astar
from planner_core.routing.point_to_point import plan_point_to_point
from planner_core.routing.snap import snap_point
from tests.assertions import assert_valid_route
from tests.fixtures.networks import FAR_AWAY, GRID_NW, GRID_SE, ISLAND

pytestmark = pytest.mark.unit

SNAP_LIMIT_M = 250.0


def node_index_of(store: GraphStore, osm_id: int) -> int:
    """Compact index of an OSM node id."""
    return int(np.flatnonzero(store.node_osm_id == osm_id)[0])


def search(store: GraphStore, source: int, target: int, **kwargs) -> SearchResult:
    """Run A* over a store with edge length as the cost."""
    return astar(
        indptr=store.indptr,
        arc_head=store.arc_head,
        arc_rev=store.arc_rev,
        arc_allowed=store.arc_allowed,
        node_lat=store.node_lat,
        node_lon=store.node_lon,
        arc_cost=store.edge_length_m[store.arc_edge],
        source=source,
        target=target,
        **kwargs,
    )


# --------------------------------------------------------------- snapping


def test_a_point_on_the_network_snaps_to_a_routable_node(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex
):
    snap = snap_point(ardennes_graph, ardennes_index, LatLon(*GRID_NW), SNAP_LIMIT_M)

    assert ardennes_graph.in_giant_component(snap.node)
    assert snap.distance_m == pytest.approx(0.0, abs=1.0)


def test_a_point_beside_the_network_snaps_and_reports_how_far_it_moved(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex
):
    # Roughly 50 m north of the grid's north-west corner.
    beside = LatLon(GRID_NW[0] + 0.00045, GRID_NW[1])

    snap = snap_point(ardennes_graph, ardennes_index, beside, SNAP_LIMIT_M)

    assert 30.0 < snap.distance_m < 70.0
    assert ardennes_graph.in_giant_component(snap.node)


def test_a_point_far_from_any_way_is_reported_with_the_nearest_one(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex
):
    # "There is no trail there" is only actionable with "but there is one here".
    stranded = LatLon(GRID_NW[0] + 0.02, GRID_NW[1])

    with pytest.raises(PointOffNetworkError) as caught:
        snap_point(ardennes_graph, ardennes_index, stranded, SNAP_LIMIT_M)

    error = caught.value
    assert error.distance_m is not None
    assert error.distance_m > SNAP_LIMIT_M
    assert error.nearest_lat is not None
    assert error.nearest_lon is not None


def test_a_point_outside_the_built_area_is_a_different_failure(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex
):
    # Distinct from off-network: this planner simply does not cover there.
    with pytest.raises(PointOutsideCoverageError):
        snap_point(
            ardennes_graph,
            ardennes_index,
            LatLon(*FAR_AWAY),
            SNAP_LIMIT_M,
            coverage=ardennes_graph.bbox(),
        )


def test_snapping_near_an_isolated_fragment_still_yields_a_routable_node(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex
):
    # The nearest way is the island, but starting there guarantees a failure
    # the rider cannot act on, so a reachable node must be returned instead.
    snap = snap_point(ardennes_graph, ardennes_index, LatLon(*ISLAND), 1e9)

    assert ardennes_graph.in_giant_component(snap.node)


# --------------------------------------------------------------- search


def test_a_route_between_opposite_corners_is_valid(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile
):
    plan = plan_point_to_point(
        ardennes_graph,
        ardennes_index,
        LatLon(*GRID_NW),
        LatLon(*GRID_SE),
        max_snap_distance_m=SNAP_LIMIT_M,
        profile=profile,
    )

    assert_valid_route(
        ardennes_graph,
        plan.route,
        start_node=plan.start_snap.node,
        end_node=plan.end_snap.node,
    )


def test_the_shortest_corner_to_corner_route_matches_the_grid_geometry(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile
):
    # Two sides of 0.001 degrees latitude plus two of longitude: about 365 m.
    plan = plan_point_to_point(
        ardennes_graph,
        ardennes_index,
        LatLon(*GRID_NW),
        LatLon(*GRID_SE),
        max_snap_distance_m=SNAP_LIMIT_M,
        profile=profile,
    )

    assert plan.route.stats.distance_m == pytest.approx(365.0, abs=2.0)
    assert plan.route.stats.n_edges == 4


def test_a_oneway_descent_cannot_be_ridden_uphill(ardennes_graph: GraphStore):
    # Way 202 runs 9 -> 3 only. Going 3 -> 9 must take the two-way column.
    store = ardennes_graph
    result = search(store, node_index_of(store, 3), node_index_of(store, 9))

    assert result.reached
    used = {int(store.edge_way_id[store.arc_edge[arc]]) for arc in result.arcs}
    assert 202 not in used


def test_the_same_descent_is_available_downhill(ardennes_graph: GraphStore):
    store = ardennes_graph
    result = search(store, node_index_of(store, 9), node_index_of(store, 3))

    assert result.reached


def test_searching_from_a_node_to_itself_is_a_trivial_success(ardennes_graph: GraphStore):
    result = search(ardennes_graph, 0, 0)

    assert result.reached
    assert result.arcs == ()
    assert result.cost == 0.0


def test_an_unreachable_target_is_reported_rather_than_searched_forever(
    ardennes_graph: GraphStore,
):
    store = ardennes_graph
    result = search(store, node_index_of(store, 11), node_index_of(store, 1))

    assert not result.reached
    assert result.arcs == ()


def test_a_settled_node_budget_stops_the_search(ardennes_graph: GraphStore):
    store = ardennes_graph
    result = search(store, node_index_of(store, 1), node_index_of(store, 9), max_settled=1)

    assert not result.reached


def test_search_is_deterministic(ardennes_graph: GraphStore):
    # A user reporting a route must be able to have it reproduced exactly.
    store = ardennes_graph
    first = search(store, node_index_of(store, 1), node_index_of(store, 9))
    second = search(store, node_index_of(store, 1), node_index_of(store, 9))

    assert first.arcs == second.arcs


def test_planning_between_coincident_points_is_rejected_clearly(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile
):
    with pytest.raises(NoRouteFoundError) as caught:
        plan_point_to_point(
            ardennes_graph,
            ardennes_index,
            LatLon(*GRID_NW),
            LatLon(*GRID_NW),
            max_snap_distance_m=SNAP_LIMIT_M,
            profile=profile,
        )

    assert caught.value.reason == "endpoints_coincide"


def test_a_route_reports_where_each_end_attached_to_the_network(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile
):
    plan = plan_point_to_point(
        ardennes_graph,
        ardennes_index,
        LatLon(GRID_NW[0] + 0.0003, GRID_NW[1]),
        LatLon(*GRID_SE),
        max_snap_distance_m=SNAP_LIMIT_M,
        profile=profile,
    )

    assert plan.start_snap.distance_m > 0.0
    assert plan.end_snap.distance_m == pytest.approx(0.0, abs=1.0)
    assert plan.settled > 0


def test_route_geometry_does_not_repeat_junction_points(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile
):
    # Each arc after the first contributes from its second point onward, or
    # every junction would appear twice in the exported track.
    plan = plan_point_to_point(
        ardennes_graph,
        ardennes_index,
        LatLon(*GRID_NW),
        LatLon(*GRID_SE),
        max_snap_distance_m=SNAP_LIMIT_M,
        profile=profile,
    )

    geometry = plan.route.geometry
    consecutive_duplicates = np.all(np.isclose(geometry[:-1], geometry[1:]), axis=1)
    assert not consecutive_duplicates.any()
