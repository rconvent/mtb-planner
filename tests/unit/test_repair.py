"""Bringing a route onto its time budget.

The safety properties matter more than the effectiveness here, because repair
splices arc sequences together and a bad splice produces a route that is not a
route. So: never return something disconnected, never return something further
from the budget than the input, never change the endpoints. Effectiveness is
asserted where the fixture network is rich enough to allow it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtb_planner.cost.costvector import materialise
from mtb_planner.cost.profiles import RideProfile
from mtb_planner.graph.store import GraphStore
from mtb_planner.routing.kernels import warm_up
from mtb_planner.routing.repair import (
    budget_error,
    fastest,
    is_connected,
    lengthen,
    repair_to_budget,
    route_nodes,
    shorten,
)
from mtb_planner.routing.router import Router

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", autouse=True)
def _compiled() -> None:
    """Compile the kernels once."""
    warm_up()


@pytest.fixture
def router(ardennes_graph: GraphStore, profile: RideProfile) -> Router:
    """A router over the fixture network."""
    return Router(ardennes_graph, materialise(ardennes_graph, profile))


def routable(store: GraphStore) -> np.ndarray:
    """Nodes in the main component."""
    return np.flatnonzero(store.node_scc == store.giant_scc)


def node_of(store: GraphStore, osm_id: int) -> int:
    """Graph index of a fixture node, by OSM id.

    Contraction renumbers nodes, so addressing them by OSM id keeps these tests
    readable and stops them breaking when the merge rule loosens.
    """
    return int(np.flatnonzero(store.node_osm_id == osm_id)[0])


@pytest.fixture
def a_path(router: Router, ardennes_graph: GraphStore) -> tuple[int, ...]:
    """The longest route the fixture network offers: five arcs, corner to steps."""
    result = router.shortest(node_of(ardennes_graph, 9), node_of(ardennes_graph, 15))

    assert result.reached
    assert len(result.arcs) >= 3
    return result.arcs


# --------------------------------------------------------------- helpers


def test_route_nodes_starts_at_the_tail_and_ends_at_the_head(ardennes_graph: GraphStore, a_path):
    nodes = route_nodes(ardennes_graph, a_path)

    assert nodes[0] == ardennes_graph.arc_tail(a_path[0])
    assert nodes[-1] == int(ardennes_graph.arc_head[a_path[-1]])
    assert len(nodes) == len(a_path) + 1


def test_an_empty_route_visits_no_nodes(ardennes_graph: GraphStore):
    assert route_nodes(ardennes_graph, []) == ()


def test_a_real_route_is_connected(ardennes_graph: GraphStore, a_path):
    assert is_connected(ardennes_graph, a_path)


def test_a_spliced_together_nonsense_route_is_not(ardennes_graph: GraphStore, a_path):
    # The check has to actually catch a break, or it protects nothing.
    broken = (a_path[0], a_path[-1]) if len(a_path) > 2 else (a_path[0], a_path[0])

    assert not is_connected(ardennes_graph, broken)


def test_the_fastest_costing_uses_time_as_its_cost(router: Router):
    quick = fastest(router.costs)

    np.testing.assert_allclose(quick.cost, router.costs.time_s)


def test_the_fastest_costing_leaves_the_reported_figures_alone(router: Router):
    # It is a different way to search, not a different ride.
    quick = fastest(router.costs)

    np.testing.assert_allclose(quick.time_s, router.costs.time_s)
    np.testing.assert_allclose(quick.distance_m, router.costs.distance_m)


def test_budget_error_is_signed(router: Router, a_path):
    elapsed = float(router.costs.time_s[np.asarray(a_path)].sum())

    assert budget_error(router.costs, a_path, elapsed * 0.5) > 0.0
    assert budget_error(router.costs, a_path, elapsed * 2.0) < 0.0
    assert budget_error(router.costs, a_path, elapsed) == pytest.approx(0.0)


# --------------------------------------------------------------- shortening


def test_shortening_a_route_already_inside_budget_leaves_it_alone(
    router: Router, ardennes_graph: GraphStore, a_path
):
    generous = float(router.costs.time_s[np.asarray(a_path)].sum()) * 2.0

    assert shorten(router, ardennes_graph, a_path, generous) == a_path


def test_shortening_never_returns_a_disconnected_route(
    router: Router, ardennes_graph: GraphStore, a_path
):
    result = shorten(router, ardennes_graph, a_path, 1.0)

    assert is_connected(ardennes_graph, result)


def test_shortening_never_moves_further_from_the_budget(
    router: Router, ardennes_graph: GraphStore, a_path
):
    elapsed = float(router.costs.time_s[np.asarray(a_path)].sum())
    target = elapsed * 0.5

    result = shorten(router, ardennes_graph, a_path, target)

    assert abs(budget_error(router.costs, result, target)) <= abs(
        budget_error(router.costs, a_path, target)
    )


def test_shortening_keeps_the_endpoints(router: Router, ardennes_graph: GraphStore, a_path):
    # It is a repair of a path between two fixed points, not a new route.
    result = shorten(router, ardennes_graph, a_path, 1.0)

    assert route_nodes(ardennes_graph, result)[0] == route_nodes(ardennes_graph, a_path)[0]
    assert route_nodes(ardennes_graph, result)[-1] == route_nodes(ardennes_graph, a_path)[-1]


def test_shortening_an_empty_route_is_harmless(router: Router, ardennes_graph: GraphStore):
    assert shorten(router, ardennes_graph, (), 3_600.0) == ()


# --------------------------------------------------------------- lengthening


def test_lengthening_a_route_already_over_budget_leaves_it_alone(
    router: Router, ardennes_graph: GraphStore, a_path
):
    assert lengthen(router, ardennes_graph, a_path, 1.0) == a_path


def test_lengthening_never_returns_a_disconnected_route(
    router: Router, ardennes_graph: GraphStore, a_path
):
    elapsed = float(router.costs.time_s[np.asarray(a_path)].sum())

    result = lengthen(router, ardennes_graph, a_path, elapsed * 3.0)

    assert is_connected(ardennes_graph, result)


def test_lengthening_keeps_the_endpoints(router: Router, ardennes_graph: GraphStore, a_path):
    elapsed = float(router.costs.time_s[np.asarray(a_path)].sum())

    result = lengthen(router, ardennes_graph, a_path, elapsed * 3.0)

    assert route_nodes(ardennes_graph, result)[0] == route_nodes(ardennes_graph, a_path)[0]
    assert route_nodes(ardennes_graph, result)[-1] == route_nodes(ardennes_graph, a_path)[-1]


def test_lengthening_never_moves_further_from_the_budget(
    router: Router, ardennes_graph: GraphStore, a_path
):
    elapsed = float(router.costs.time_s[np.asarray(a_path)].sum())
    target = elapsed * 3.0

    result = lengthen(router, ardennes_graph, a_path, target)

    assert abs(budget_error(router.costs, result, target)) <= abs(
        budget_error(router.costs, a_path, target)
    )


def test_lengthening_a_route_that_can_grow_does_grow(
    router: Router, ardennes_graph: GraphStore, a_path
):
    # The fixture grid has parallel routes, so there is somewhere to detour.
    elapsed = float(router.costs.time_s[np.asarray(a_path)].sum())

    result = lengthen(router, ardennes_graph, a_path, elapsed * 2.0)

    assert float(router.costs.time_s[np.asarray(result)].sum()) > elapsed


def test_lengthening_an_empty_route_is_harmless(router: Router, ardennes_graph: GraphStore):
    assert lengthen(router, ardennes_graph, (), 3_600.0) == ()


# --------------------------------------------------------------- repair


def test_a_route_already_within_tolerance_is_untouched(
    router: Router, ardennes_graph: GraphStore, a_path
):
    elapsed = float(router.costs.time_s[np.asarray(a_path)].sum())

    assert repair_to_budget(router, ardennes_graph, a_path, elapsed, 0.10) == a_path


def test_repair_chooses_the_right_direction(router: Router, ardennes_graph: GraphStore, a_path):
    elapsed = float(router.costs.time_s[np.asarray(a_path)].sum())

    longer = repair_to_budget(router, ardennes_graph, a_path, elapsed * 2.5, 0.10)
    assert float(router.costs.time_s[np.asarray(longer)].sum()) >= elapsed


def test_repair_never_returns_a_disconnected_route(
    router: Router, ardennes_graph: GraphStore, a_path
):
    for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
        elapsed = float(router.costs.time_s[np.asarray(a_path)].sum())
        result = repair_to_budget(router, ardennes_graph, a_path, elapsed * factor, 0.10)

        assert is_connected(ardennes_graph, result)


def test_repair_is_deterministic(router: Router, ardennes_graph: GraphStore, a_path):
    # No RNG anywhere in the pipeline, so this holds without a seed.
    elapsed = float(router.costs.time_s[np.asarray(a_path)].sum())

    first = repair_to_budget(router, ardennes_graph, a_path, elapsed * 2.0, 0.10)
    second = repair_to_budget(router, ardennes_graph, a_path, elapsed * 2.0, 0.10)

    assert first == second


def test_repair_terminates_on_an_impossible_budget(
    router: Router, ardennes_graph: GraphStore, a_path
):
    # A seventeen-node network cannot fill a day. Repair must give up, not spin.
    result = repair_to_budget(router, ardennes_graph, a_path, 86_400.0, 0.10)

    assert is_connected(ardennes_graph, result)


def test_repair_of_an_empty_route_is_harmless(router: Router, ardennes_graph: GraphStore):
    assert repair_to_budget(router, ardennes_graph, (), 3_600.0, 0.10) == ()


def test_a_nonsensical_budget_is_not_acted_on(router: Router, ardennes_graph: GraphStore, a_path):
    assert repair_to_budget(router, ardennes_graph, a_path, 0.0, 0.10) == a_path
