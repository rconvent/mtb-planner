"""The wall-clock budget on a search.

Numba cannot read a clock, so the timeout is enforced by running the search
under a settle ceiling and raising it between rounds. Two things must hold: a
search that finishes inside the first ceiling must pay nothing for the
machinery, and one that cannot finish must be reported as abandoned rather than
as "no such route" — the two need different answers from the rider.

The ceiling is injected in these tests because the fixture network has
seventeen nodes and the production ceiling is a quarter of a million.
"""

from __future__ import annotations

import numpy as np
import pytest

from planner_core.core.errors import NoRouteFoundError
from planner_core.core.geo import LatLon
from planner_core.cost.costvector import materialise
from planner_core.cost.profiles import RideProfile
from planner_core.graph.spatial_index import SpatialIndex
from planner_core.graph.store import GraphStore
from planner_core.routing.budget import INITIAL_SETTLE_CEILING, SETTLE_GROWTH, Deadline
from planner_core.routing.kernels import warm_up
from planner_core.routing.point_to_point import plan_point_to_point
from planner_core.routing.router import Router
from tests.fixtures.networks import GRID_NW, GRID_SE

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", autouse=True)
def _compiled() -> None:
    """Compile once, so nothing here is measuring the compiler."""
    warm_up()


@pytest.fixture
def router(ardennes_graph: GraphStore, profile: RideProfile) -> Router:
    """A router over the fixture network."""
    return Router(ardennes_graph, materialise(ardennes_graph, profile))


def routable(store: GraphStore) -> np.ndarray:
    """Nodes in the main component, which is where searches can succeed."""
    return np.flatnonzero(store.node_scc == store.giant_scc)


# --------------------------------------------------------------- the deadline


def test_a_fresh_deadline_has_time_left():
    deadline = Deadline.after(60.0)

    assert not deadline.expired
    assert 0.0 < deadline.remaining_s <= 60.0


def test_a_zero_budget_is_already_spent():
    assert Deadline.after(0.0).expired


def test_remaining_time_never_goes_negative():
    # Callers scale work by what is left; a negative would invert that.
    assert Deadline.after(-5.0).remaining_s == 0.0


def test_a_deadline_is_immutable():
    deadline = Deadline.after(60.0)

    with pytest.raises(AttributeError):
        deadline.expires_at = 0.0


def test_the_deepening_ceiling_is_bigger_than_a_country_sized_graph():
    # Luxembourg is 164,650 nodes. If the first ceiling did not clear a whole
    # extract, every long search would restart for nothing.
    assert INITIAL_SETTLE_CEILING > 164_650
    assert SETTLE_GROWTH > 1


# --------------------------------------------------------------- the search


def test_a_search_that_fits_the_first_ceiling_is_answered_normally(
    router: Router, ardennes_graph: GraphStore
):
    nodes = routable(ardennes_graph)

    result = router.shortest_within(int(nodes[0]), int(nodes[-1]), deadline=Deadline.after(60.0))

    assert result.reached
    assert not result.truncated


def test_a_budgeted_search_finds_the_same_route_as_an_unbudgeted_one(
    router: Router, ardennes_graph: GraphStore
):
    # The budget is a ceiling, not a different algorithm.
    nodes = routable(ardennes_graph)
    source, target = int(nodes[0]), int(nodes[-1])

    plain = router.shortest(source, target)
    budgeted = router.shortest_within(source, target, deadline=Deadline.after(60.0))

    assert budgeted.arcs == plain.arcs
    assert budgeted.cost == pytest.approx(plain.cost)


def test_no_deadline_means_no_ceiling(router: Router, ardennes_graph: GraphStore):
    nodes = routable(ardennes_graph)

    result = router.shortest_within(int(nodes[0]), int(nodes[-1]), deadline=None)

    assert result.reached
    assert not result.truncated


def test_an_exhausted_budget_abandons_the_search_rather_than_denying_the_route(
    router: Router, ardennes_graph: GraphStore
):
    # One settled node is nowhere near enough, and the budget is already spent,
    # so there is no second round. The route exists; we simply stopped looking.
    nodes = routable(ardennes_graph)

    result = router.shortest_within(
        int(nodes[0]),
        int(nodes[-1]),
        deadline=Deadline.after(0.0),
        initial_ceiling=1,
    )

    assert not result.reached
    assert result.truncated


def test_a_search_deepens_until_it_succeeds_when_there_is_time(
    router: Router, ardennes_graph: GraphStore
):
    # Starting from a ceiling of one, the loop must raise it rather than give
    # up — and land on the same route an unbounded search finds.
    nodes = routable(ardennes_graph)
    source, target = int(nodes[0]), int(nodes[-1])

    result = router.shortest_within(
        source, target, deadline=Deadline.after(60.0), initial_ceiling=1
    )

    assert result.reached
    assert not result.truncated
    assert result.arcs == router.shortest(source, target).arcs


def test_an_unreachable_target_is_final_not_truncated(router: Router, ardennes_graph: GraphStore):
    # The isolated fragment cannot be reached however long we look. Reporting
    # that as "ran out of budget" would send the client back to retry forever.
    island = int(np.flatnonzero(ardennes_graph.node_osm_id == 11)[0])
    mainland = int(routable(ardennes_graph)[0])

    result = router.shortest_within(island, mainland, deadline=Deadline.after(60.0))

    assert not result.reached
    assert not result.truncated


def test_a_ceiling_that_stops_the_search_is_reported_as_truncation(
    router: Router, ardennes_graph: GraphStore
):
    # `shortest` is the honest primitive: it says whether its ceiling was what
    # stopped it, which is what the deepening loop reads.
    nodes = routable(ardennes_graph)

    result = router.shortest(int(nodes[0]), int(nodes[-1]), max_settled=1)

    assert not result.reached
    assert result.truncated


def test_a_completed_search_is_never_flagged_as_truncated(
    router: Router, ardennes_graph: GraphStore
):
    nodes = routable(ardennes_graph)

    result = router.shortest(int(nodes[0]), int(nodes[-1]), max_settled=10_000)

    assert result.reached
    assert not result.truncated


# --------------------------------------------------------------- planning


def test_a_plan_that_runs_out_of_budget_says_so_rather_than_denying_the_route(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile: RideProfile
):
    # "Disconnected" would be a lie — these corners are connected — and it
    # would send the rider looking for a different start instead of retrying.
    with pytest.raises(NoRouteFoundError) as caught:
        plan_point_to_point(
            ardennes_graph,
            ardennes_index,
            LatLon(*GRID_NW),
            LatLon(*GRID_SE),
            max_snap_distance_m=250.0,
            profile=profile,
            max_settled=1,
        )

    assert caught.value.reason == "search_timeout"


def test_a_plan_with_room_to_search_is_unaffected_by_the_budget(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile: RideProfile
):
    plan = plan_point_to_point(
        ardennes_graph,
        ardennes_index,
        LatLon(*GRID_NW),
        LatLon(*GRID_SE),
        max_snap_distance_m=250.0,
        profile=profile,
        deadline=Deadline.after(60.0),
    )

    assert plan.route.stats.distance_m > 0
    assert plan.settled >= 1
