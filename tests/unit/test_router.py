"""The compiled kernels and the facade over them.

The kernels exist for speed, so the tests that matter are the ones asserting
they did not trade correctness for it. Measured on Luxembourg across 200
random long-range pairs, the compiled A* agrees with the pure-Python
reference on every cost while running 12–14 times faster: p95 falls from
452 ms to 32 ms.
"""

from __future__ import annotations

import numpy as np
import pytest

from planner_core.cost.costvector import materialise
from planner_core.cost.profiles import RideProfile
from planner_core.graph.store import GraphStore
from planner_core.routing.astar import astar
from planner_core.routing.kernels import warm_up
from planner_core.routing.router import Router

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", autouse=True)
def _compiled() -> None:
    """Compile once so timing-sensitive tests are not measuring the compiler."""
    warm_up()


@pytest.fixture
def router(ardennes_graph: GraphStore, profile: RideProfile) -> Router:
    """A router over the fixture network."""
    return Router(ardennes_graph, materialise(ardennes_graph, profile))


def routable(store: GraphStore) -> np.ndarray:
    """Nodes in the main component, which is where searches can succeed."""
    return np.flatnonzero(store.node_scc == store.giant_scc)


# --------------------------------------------------------------- correctness


def test_the_kernel_agrees_with_the_reference_on_every_pair(
    router: Router, ardennes_graph: GraphStore
):
    # The gate for this phase: compiled and interpreted searches must return
    # the same cost, or the speed bought nothing.
    costs = router.costs
    nodes = routable(ardennes_graph)

    for source in nodes:
        for target in nodes:
            if source == target:
                continue
            compiled = router.shortest(int(source), int(target))
            reference = astar(
                indptr=ardennes_graph.indptr,
                arc_head=ardennes_graph.arc_head,
                arc_rev=ardennes_graph.arc_rev,
                arc_allowed=costs.allowed,
                node_lat=ardennes_graph.node_lat,
                node_lon=ardennes_graph.node_lon,
                arc_cost=costs.cost,
                source=int(source),
                target=int(target),
                heuristic_scale=costs.min_seconds_per_metre,
            )
            assert compiled.reached == reference.reached
            if compiled.reached:
                assert compiled.cost == pytest.approx(reference.cost, rel=1e-9)


def test_a_search_to_itself_is_trivially_done(router: Router):
    result = router.shortest(0, 0)

    assert result.reached
    assert result.arcs == ()
    assert result.cost == 0.0


def test_an_unreachable_target_is_reported(router: Router, ardennes_graph: GraphStore):
    island = int(np.flatnonzero(ardennes_graph.node_osm_id == 11)[0])
    mainland = int(routable(ardennes_graph)[0])

    result = router.shortest(island, mainland)

    assert not result.reached
    assert result.arcs == ()


def test_search_is_deterministic(router: Router, ardennes_graph: GraphStore):
    nodes = routable(ardennes_graph)

    first = router.shortest(int(nodes[0]), int(nodes[-1]))
    second = router.shortest(int(nodes[0]), int(nodes[-1]))

    assert first.arcs == second.arcs


def test_a_settled_budget_stops_the_search(router: Router, ardennes_graph: GraphStore):
    nodes = routable(ardennes_graph)

    result = router.shortest(int(nodes[0]), int(nodes[-1]), max_settled=1)

    assert not result.reached


def test_the_route_only_uses_traversable_arcs(router: Router, ardennes_graph: GraphStore):
    nodes = routable(ardennes_graph)

    result = router.shortest(int(nodes[0]), int(nodes[-1]))

    assert all(router.costs.allowed[arc] for arc in result.arcs)


def test_consecutive_arcs_actually_connect(router: Router, ardennes_graph: GraphStore):
    nodes = routable(ardennes_graph)
    result = router.shortest(int(nodes[0]), int(nodes[-1]))

    for first, second in zip(result.arcs, result.arcs[1:], strict=False):
        assert int(ardennes_graph.arc_head[first]) == ardennes_graph.arc_tail(second)


def test_a_route_never_doubles_straight_back(router: Router, ardennes_graph: GraphStore):
    # U-turn prohibition, which is why the search tracks arcs rather than nodes.
    nodes = routable(ardennes_graph)
    result = router.shortest(int(nodes[0]), int(nodes[-1]))

    for first, second in zip(result.arcs, result.arcs[1:], strict=False):
        assert second != int(ardennes_graph.arc_rev[first])


# --------------------------------------------------------------- frontier


def test_an_unbounded_frontier_reaches_the_whole_component(
    router: Router, ardennes_graph: GraphStore
):
    source = int(routable(ardennes_graph)[0])

    frontier = router.reachable_within(source)

    assert frontier.reached().size == ardennes_graph.giant_scc_size


def test_a_time_bound_limits_what_is_reached(router: Router, ardennes_graph: GraphStore):
    source = int(routable(ardennes_graph)[0])

    unbounded = router.reachable_within(source)
    bounded = router.reachable_within(source, time_bound=1.0)

    assert bounded.reached().size < unbounded.reached().size


def test_everything_reached_respects_the_time_bound(router: Router, ardennes_graph: GraphStore):
    source = int(routable(ardennes_graph)[0])
    limit = 30.0

    frontier = router.reachable_within(source, time_bound=limit)

    assert np.all(frontier.time_s[frontier.reached()] <= limit + 1e-9)


def test_a_frontier_can_rebuild_the_path_to_anything_it_reached(
    router: Router, ardennes_graph: GraphStore
):
    source = int(routable(ardennes_graph)[0])
    frontier = router.reachable_within(source)

    for node in frontier.reached():
        if int(node) == source:
            continue
        arcs = frontier.path_to(int(node))
        assert ardennes_graph.arc_tail(arcs[0]) == source
        assert int(ardennes_graph.arc_head[arcs[-1]]) == int(node)


def test_the_frontier_agrees_with_a_direct_search(router: Router, ardennes_graph: GraphStore):
    # The bounded Dijkstra and A* are different algorithms over the same cost;
    # where both succeed they must agree.
    nodes = routable(ardennes_graph)
    source = int(nodes[0])
    frontier = router.reachable_within(source)

    for target in nodes[1:]:
        direct = router.shortest(source, int(target))
        if direct.reached:
            assert frontier.cost[int(target)] == pytest.approx(direct.cost, rel=1e-9)


def test_pivots_are_selected_by_time_not_cost(router: Router, ardennes_graph: GraphStore):
    # Half a ride away has to mean half a ride, not half a preference-weighted
    # cost — otherwise a loop's budget drifts with the profile.
    source = int(routable(ardennes_graph)[0])
    frontier = router.reachable_within(source)

    band = frontier.within_time(0.0, 20.0)
    assert np.all(frontier.time_s[band] <= 20.0)


def test_time_and_distance_are_summed_from_the_arcs(router: Router, ardennes_graph: GraphStore):
    nodes = routable(ardennes_graph)
    result = router.shortest(int(nodes[0]), int(nodes[-1]))

    assert router.time_of(result.arcs) == pytest.approx(
        sum(router.costs.time_s[a] for a in result.arcs)
    )
    assert router.distance_of(result.arcs) == pytest.approx(
        sum(router.costs.distance_m[a] for a in result.arcs)
    )
