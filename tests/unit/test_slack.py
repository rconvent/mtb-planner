"""Spending the slack in a point-to-point time budget.

The property that matters is not which route comes back but that the budget is
never exceeded and never ignored: more time must buy ground the rider prefers,
and a route that does not fit must say so rather than being quietly trimmed to
something that does.

The lattice is the network used throughout. `ardennes_like` has too few ways
between any two points to offer the planner a choice, and a budget can only be
spent where there is an alternative to spend it on.
"""

from __future__ import annotations

import numpy as np
import pytest

from planner_core.core.geo import LatLon
from planner_core.cost.costvector import materialise
from planner_core.cost.profiles import MAX_DETOUR, RideProfile
from planner_core.graph.spatial_index import SpatialIndex
from planner_core.graph.store import GraphStore
from planner_core.routing.point_to_point import PointToPointPlan, plan_point_to_point
from planner_core.routing.router import Router
from planner_core.routing.slack import spend_slack
from planner_core.routing.snap import snap_point
from tests.fixtures.networks import LATTICE_NE, LATTICE_SW

pytestmark = pytest.mark.unit

SNAP_LIMIT_M = 250.0

#: A budget no fixture route can exceed, for the cases about what a generous
#: budget buys rather than about what a tight one refuses.
GENEROUS_S = 24 * 3600.0


def plan_across(
    store: GraphStore,
    index: SpatialIndex,
    profile: RideProfile,
    budget_s: float | None,
) -> PointToPointPlan:
    """Plan corner to corner across the lattice under a budget."""
    return plan_point_to_point(
        store,
        index,
        LatLon(*LATTICE_SW),
        LatLon(*LATTICE_NE),
        max_snap_distance_m=SNAP_LIMIT_M,
        profile=profile,
        budget_s=budget_s,
    )


def corners(store: GraphStore, index: SpatialIndex) -> tuple[int, int]:
    """The two endpoints as node indices, for calling the search directly."""
    start = snap_point(store, index, LatLon(*LATTICE_SW), SNAP_LIMIT_M)
    end = snap_point(store, index, LatLon(*LATTICE_NE), SNAP_LIMIT_M)
    return start.node, end.node


def test_no_budget_leaves_the_profile_in_charge(
    lattice_graph: GraphStore, lattice_index: SpatialIndex, profile: RideProfile
):
    plan = plan_across(lattice_graph, lattice_index, profile, None)

    assert plan.detour == pytest.approx(profile.detour)
    assert plan.within_budget is None


def test_a_generous_budget_never_returns_a_route_that_misses_it(
    lattice_graph: GraphStore, lattice_index: SpatialIndex, profile: RideProfile
):
    plan = plan_across(lattice_graph, lattice_index, profile, GENEROUS_S)

    assert plan.within_budget is True
    assert plan.route.stats.moving_time_s <= GENEROUS_S


def test_slack_is_spent_on_preference_not_hoarded(
    lattice_graph: GraphStore, lattice_index: SpatialIndex, profile: RideProfile
):
    # The whole point of a budget on a route that has one answer: with time to
    # spare the planner should be readier to detour than the profile alone is.
    tight = plan_across(lattice_graph, lattice_index, profile, None)
    generous = plan_across(lattice_graph, lattice_index, profile, GENEROUS_S)

    assert generous.detour > tight.detour
    assert generous.detour <= MAX_DETOUR


def test_a_budget_below_the_quickest_route_is_reported_not_enforced(
    lattice_graph: GraphStore, lattice_index: SpatialIndex, profile: RideProfile
):
    # A minute is less than any route across the lattice takes. The rider still
    # gets a route to where they asked to go, and is told it runs long.
    plan = plan_across(lattice_graph, lattice_index, profile, 60.0)

    assert plan.within_budget is False
    assert plan.detour == pytest.approx(profile.detour)
    assert plan.route.stats.moving_time_s > 60.0


@pytest.mark.parametrize("factor", [1.0, 1.25, 1.5, 2.0, 4.0])
def test_every_probe_stays_inside_the_budget_it_was_given(
    lattice_graph: GraphStore, lattice_index: SpatialIndex, profile: RideProfile, factor: float
):
    # Sweeps the interesting range: the bisection is only correct if no budget
    # between the profile's own route and the greediest one comes back over.
    quickest = plan_across(lattice_graph, lattice_index, profile, None)
    budget_s = quickest.route.stats.moving_time_s * factor

    plan = plan_across(lattice_graph, lattice_index, profile, budget_s)

    assert plan.within_budget is True
    assert plan.route.stats.moving_time_s <= budget_s + 1e-6


def test_the_time_reported_is_summed_from_the_arcs_not_read_off_the_cost(
    lattice_graph: GraphStore, lattice_index: SpatialIndex, profile: RideProfile
):
    # The separation invariant, at the one place a raised lambda could break it:
    # `cost` changes under the search, and time, distance and preference must
    # not move with it.
    costs = materialise(lattice_graph, profile)
    source, target = corners(lattice_graph, lattice_index)
    baseline = Router(lattice_graph, costs).shortest(source, target)

    spent = spend_slack(
        lattice_graph,
        costs,
        source,
        target,
        budget_s=GENEROUS_S,
        floor=profile.detour,
        ceiling=MAX_DETOUR,
        baseline=baseline,
    )

    np.testing.assert_allclose(spent.costs.time_s, costs.time_s)
    np.testing.assert_allclose(spent.costs.distance_m, costs.distance_m)
    np.testing.assert_allclose(spent.costs.preference, costs.preference)
    assert spent.time_s == pytest.approx(
        float(costs.time_s[list(spent.result.arcs)].sum()), rel=1e-9
    )


def test_a_ceiling_at_the_floor_leaves_nothing_to_search(
    lattice_graph: GraphStore, lattice_index: SpatialIndex, profile: RideProfile
):
    costs = materialise(lattice_graph, profile)
    source, target = corners(lattice_graph, lattice_index)
    baseline = Router(lattice_graph, costs).shortest(source, target)

    spent = spend_slack(
        lattice_graph,
        costs,
        source,
        target,
        budget_s=GENEROUS_S,
        floor=profile.detour,
        ceiling=profile.detour,
        baseline=baseline,
    )

    assert spent.probes == 1
    assert spent.detour == pytest.approx(profile.detour)
    assert spent.result.arcs == baseline.arcs
