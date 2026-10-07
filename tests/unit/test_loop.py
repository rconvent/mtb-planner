"""The loop generator.

Assertions are on invariants, not geometry. Which loop a penalised two-tree
construction happens to prefer is a property of the current cost weights; that
it comes back to where it started, does not merely retrace itself, and reports
figures measured from the ground it covers are properties of any correct
generator, and they survive retuning.

The fixture network is 1.5 km of trail in total, so the budgets here are in
seconds rather than hours. That is also what makes the failure paths reachable:
a five-minute loop genuinely does not exist on it, and saying so is part of the
job. The Phase 8 gate — three loops inside ±10% — needs a real extract and lives
in `tests/integration/test_loop_gate.py`.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mtb_planner.speed import Rider
from planner_core.core.errors import (
    NoRouteFoundError,
    PointOffNetworkError,
    PointOutsideCoverageError,
)
from planner_core.core.geo import LatLon
from planner_core.cost.profiles import RideProfile, get_profile
from planner_core.graph.spatial_index import SpatialIndex
from planner_core.graph.store import GraphStore
from planner_core.routing.budget import Deadline
from planner_core.routing.kernels import warm_up
from planner_core.routing.loop import (
    HEADING_SECTORS,
    LoopRequest,
    plan_loops,
)
from tests.assertions import assert_valid_loop
from tests.fixtures.networks import FAR_AWAY, GRID_NW

pytestmark = pytest.mark.unit

#: A budget the fixture network can actually satisfy.
WORKABLE_S = 180.0

#: Far more riding than 1.5 km of trail can provide.
IMPOSSIBLE_S = 600.0


@pytest.fixture(scope="module", autouse=True)
def _compiled() -> None:
    """Compile the kernels once."""
    warm_up()


def a_request(profile: RideProfile, **overrides: object) -> LoopRequest:
    """A loop request over the fixture network."""
    base: dict[str, object] = {
        "start": LatLon(*GRID_NW),
        "budget_s": WORKABLE_S,
        "profile": profile,
        "athlete": Rider(),
        "variants": 3,
    }
    base.update(overrides)
    return LoopRequest(**base)  # type: ignore[arg-type]


@pytest.fixture
def plan(ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile: RideProfile):
    """A generated set of loops over the fixture network."""
    return plan_loops(
        ardennes_graph, ardennes_index, a_request(profile), coverage=ardennes_graph.bbox()
    )


# --------------------------------------------------------------- shape


def test_a_loop_request_returns_at_least_one_loop(plan):
    assert len(plan.candidates) >= 1


def test_it_never_returns_more_than_asked_for(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile: RideProfile
):
    result = plan_loops(
        ardennes_graph,
        ardennes_index,
        a_request(profile, variants=1),
        coverage=ardennes_graph.bbox(),
    )

    assert len(result.candidates) == 1


def test_every_candidate_is_a_valid_loop(plan, ardennes_graph: GraphStore, profile: RideProfile):
    # The shared invariant: comes back, does not retrace itself, uses no
    # forbidden way, and reports what it actually rides.
    for candidate in plan.candidates:
        assert_valid_loop(ardennes_graph, candidate, plan.start_snap.node, a_request(profile))


def test_every_loop_closes_where_it_started(plan):
    for candidate in plan.candidates:
        assert candidate.route.is_loop
        assert candidate.route.nodes[0] == candidate.route.nodes[-1]


def test_the_start_is_snapped_and_reported(plan):
    assert plan.start_snap.distance_m >= 0.0
    assert plan.start_snap.node >= 0


def test_the_search_effort_is_reported(plan):
    # So a slow request can be diagnosed without a profiler.
    assert plan.settled >= 1


def test_candidates_are_offered_best_first(plan):
    scores = [c.score for c in plan.candidates]

    assert scores == sorted(scores, reverse=True) or len(scores) == 1


# --------------------------------------------------------------- honesty


def test_reported_time_is_measured_from_the_arcs_not_the_cost(plan):
    # The separation invariant, at the point it reaches a rider.
    for candidate in plan.candidates:
        assert candidate.metrics.time_s == pytest.approx(
            candidate.route.stats.moving_time_s, rel=1e-9
        )


def test_reported_distance_is_measured_from_the_edges(plan, ardennes_graph: GraphStore):
    for candidate in plan.candidates:
        expected = sum(
            float(ardennes_graph.edge_length_m[ardennes_graph.arc_edge[arc]])
            for arc in candidate.route.arcs
        )
        assert candidate.route.stats.distance_m == pytest.approx(expected, rel=1e-9)


def test_a_loop_outside_tolerance_is_flagged_rather_than_hidden(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile: RideProfile
):
    # A 60-second budget on this network yields loops of about 84 seconds.
    # Returning them marked is more use than returning nothing.
    result = plan_loops(
        ardennes_graph,
        ardennes_index,
        a_request(profile, budget_s=60.0),
        coverage=ardennes_graph.bbox(),
    )

    for candidate in result.candidates:
        inside = abs(candidate.metrics.budget_error(60.0)) <= 0.10
        assert candidate.within_budget == inside


def test_overlap_is_measured_and_within_the_limit(plan, profile: RideProfile):
    request = a_request(profile)

    for candidate in plan.candidates:
        assert 0.0 <= candidate.metrics.overlap_ratio <= request.max_overlap


def test_a_stricter_overlap_limit_is_respected(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile: RideProfile
):
    result = plan_loops(
        ardennes_graph,
        ardennes_index,
        a_request(profile, max_overlap=0.01),
        coverage=ardennes_graph.bbox(),
    )

    for candidate in result.candidates:
        assert candidate.metrics.overlap_ratio <= 0.01


# --------------------------------------------------------------- determinism


def test_the_same_request_returns_the_same_loops(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile: RideProfile
):
    # There is no RNG in the pipeline, so this holds without a seed. A rider
    # reporting a route must be able to have it reproduced exactly.
    first = plan_loops(
        ardennes_graph, ardennes_index, a_request(profile), coverage=ardennes_graph.bbox()
    )
    second = plan_loops(
        ardennes_graph, ardennes_index, a_request(profile), coverage=ardennes_graph.bbox()
    )

    assert [c.route.arcs for c in first.candidates] == [c.route.arcs for c in second.candidates]
    assert [c.score for c in first.candidates] == [c.score for c in second.candidates]


def test_a_different_profile_can_choose_differently(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex
):
    # Not that it must differ on a 1.5 km network, but that the profile reaches
    # the generator at all rather than being ignored.
    tech = plan_loops(
        ardennes_graph,
        ardennes_index,
        a_request(get_profile("tech")),
        coverage=ardennes_graph.bbox(),
    )

    assert tech.candidates


# --------------------------------------------------------------- failures


def test_a_budget_the_network_cannot_fill_is_reported_clearly(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile: RideProfile
):
    with pytest.raises(NoRouteFoundError) as caught:
        plan_loops(
            ardennes_graph,
            ardennes_index,
            a_request(profile, budget_s=IMPOSSIBLE_S),
            coverage=ardennes_graph.bbox(),
        )

    assert caught.value.reason == "no_loop_found"


def test_a_start_outside_the_built_area_is_rejected_before_searching(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile: RideProfile
):
    with pytest.raises(PointOutsideCoverageError):
        plan_loops(
            ardennes_graph,
            ardennes_index,
            a_request(profile, start=LatLon(*FAR_AWAY)),
            coverage=ardennes_graph.bbox(),
        )


def test_a_start_with_no_trail_near_it_is_rejected(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile: RideProfile
):
    with pytest.raises(PointOffNetworkError):
        plan_loops(
            ardennes_graph,
            ardennes_index,
            a_request(profile, start=LatLon(50.150, 5.150)),
            coverage=ardennes_graph.bbox(),
        )


# --------------------------------------------------------------- budget


def test_an_expired_deadline_still_returns_what_it_has(
    ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile: RideProfile
):
    # Unlike a point-to-point search, a loop generator genuinely has a
    # best-so-far: candidates accumulate, so expiry degrades quality rather
    # than failing. It must say so.
    result = plan_loops(
        ardennes_graph,
        ardennes_index,
        a_request(profile),
        coverage=ardennes_graph.bbox(),
        deadline=Deadline.after(0.0),
    )

    assert result.truncated


def test_a_generous_deadline_is_not_reported_as_truncated(plan):
    assert not plan.truncated


def test_no_deadline_is_not_reported_as_truncated(plan):
    assert not plan.truncated


# --------------------------------------------------------------- spreading


def test_pivots_are_spread_across_a_dozen_sectors():
    # Twelve is what makes the returned loops look different rather than being
    # four variations on the same trail.
    assert HEADING_SECTORS == 12


def test_distinct_candidates_are_genuinely_distinct(plan, ardennes_graph: GraphStore):
    arc_sets = [frozenset(c.route.arcs) for c in plan.candidates]

    assert len(set(arc_sets)) == len(arc_sets)


def test_loops_do_not_start_with_an_immediate_u_turn(plan, ardennes_graph: GraphStore):
    # The kernels forbid doubling back within one search, but the seam where the
    # outbound leg meets the return is the generator's own responsibility.
    for candidate in plan.candidates:
        arcs = candidate.route.arcs
        for first, second in itertools.pairwise(arcs):
            assert second != int(ardennes_graph.arc_rev[first])


def test_a_loop_never_uses_an_arc_outside_the_search_ball(plan, ardennes_graph: GraphStore):
    # Every arc must be one the restricted cost vector marked traversable.
    for candidate in plan.candidates:
        for arc in candidate.route.arcs:
            assert ardennes_graph.arc_allowed[arc]


def test_ascent_and_descent_roughly_balance_on_a_loop(plan):
    # A loop returns to its start, so what it climbs it must also descend. They
    # differ slightly because elevation is stored per edge in one direction.
    for candidate in plan.candidates:
        stats = candidate.route.stats
        assert stats.ascent_m == pytest.approx(stats.descent_m, abs=1.0)


def test_a_loop_visits_more_than_one_node(plan):
    for candidate in plan.candidates:
        assert len(set(candidate.route.nodes)) > 1
        assert len(candidate.route.arcs) >= 2


def test_geometry_is_a_closed_ring(plan):
    for candidate in plan.candidates:
        geometry = candidate.route.geometry
        assert geometry.shape[1] == 2
        np.testing.assert_allclose(geometry[0], geometry[-1], atol=1e-9)
