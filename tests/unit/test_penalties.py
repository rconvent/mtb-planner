"""Penalising a corridor so the return leg finds another one.

The property that matters most is the one that is easiest to break by accident:
a penalty must move `cost` and nothing else. The moment it reaches `time_s`,
every duration reported for a loop is inflated by however much backtracking the
generator happened to discourage, and the separation invariant the whole cost
model rests on is gone.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtb_planner.cost.costvector import materialise
from mtb_planner.cost.profiles import RideProfile
from mtb_planner.graph.store import GraphStore
from mtb_planner.routing.penalties import RETURN_PENALTY, penalise

pytestmark = pytest.mark.unit


@pytest.fixture
def costs(ardennes_graph: GraphStore, profile: RideProfile):
    """A cost vector over the fixture network."""
    return materialise(ardennes_graph, profile)


def test_a_penalised_arc_costs_more(ardennes_graph: GraphStore, costs):
    penalised = penalise(costs, ardennes_graph, [0])

    assert penalised.cost[0] == pytest.approx(costs.cost[0] * RETURN_PENALTY)


def test_the_reverse_twin_is_penalised_too(ardennes_graph: GraphStore, costs):
    # Otherwise the return leg simply rides the outbound trail backwards, which
    # is the same trail as far as the rider is concerned.
    twin = int(ardennes_graph.arc_rev[0])

    penalised = penalise(costs, ardennes_graph, [0])

    assert penalised.cost[twin] == pytest.approx(costs.cost[twin] * RETURN_PENALTY)


def test_untouched_arcs_keep_their_cost(ardennes_graph: GraphStore, costs):
    penalised = penalise(costs, ardennes_graph, [0])
    touched = {0, int(ardennes_graph.arc_rev[0])}
    others = [a for a in range(ardennes_graph.n_arcs) if a not in touched]

    np.testing.assert_allclose(penalised.cost[others], costs.cost[others])


def test_time_and_distance_are_never_penalised(ardennes_graph: GraphStore, costs):
    # The separation invariant. A penalty is a routing preference, not a claim
    # about how long the ride takes.
    penalised = penalise(costs, ardennes_graph, range(ardennes_graph.n_arcs))

    np.testing.assert_allclose(penalised.time_s, costs.time_s)
    np.testing.assert_allclose(penalised.distance_m, costs.distance_m)


def test_preference_is_never_penalised(ardennes_graph: GraphStore, costs):
    # Scoring reads preference, so a penalty leaking into it would make a route
    # look worse for having been discouraged.
    penalised = penalise(costs, ardennes_graph, range(ardennes_graph.n_arcs))

    np.testing.assert_allclose(penalised.preference, costs.preference)


def test_the_original_cost_vector_is_left_alone(ardennes_graph: GraphStore, costs):
    # Pivots are penalised one at a time against the same base vector; mutating
    # it would make each candidate inherit the last one's penalties.
    before = costs.cost.copy()

    penalise(costs, ardennes_graph, [0, 1, 2])

    np.testing.assert_allclose(costs.cost, before)


def test_penalising_nothing_changes_nothing(ardennes_graph: GraphStore, costs):
    penalised = penalise(costs, ardennes_graph, [])

    np.testing.assert_allclose(penalised.cost, costs.cost)
    np.testing.assert_array_equal(penalised.allowed, costs.allowed)


def test_the_heuristic_stays_admissible_after_penalising(ardennes_graph: GraphStore, costs):
    # A* correctness depends on cost never dropping below distance times the
    # fastest second per metre. Penalties only ever raise cost, and they leave
    # the arrays the scale is derived from untouched.
    penalised = penalise(costs, ardennes_graph, range(ardennes_graph.n_arcs))

    assert penalised.min_seconds_per_metre == pytest.approx(costs.min_seconds_per_metre)
    assert np.all(penalised.cost >= costs.cost - 1e-9)


def test_a_forbidden_arc_becomes_untraversable(ardennes_graph: GraphStore, costs):
    # How the seam at a loop's turning point is stopped from being a U-turn.
    penalised = penalise(costs, ardennes_graph, [], forbid=[3])

    assert penalised.allowed[3] == 0


def test_forbidding_one_arc_leaves_the_rest_traversable(ardennes_graph: GraphStore, costs):
    penalised = penalise(costs, ardennes_graph, [], forbid=[3])
    others = [a for a in range(ardennes_graph.n_arcs) if a != 3]

    np.testing.assert_array_equal(penalised.allowed[others], costs.allowed[others])


def test_a_heavier_factor_costs_more(ardennes_graph: GraphStore, costs):
    light = penalise(costs, ardennes_graph, [0], factor=2.0)
    heavy = penalise(costs, ardennes_graph, [0], factor=8.0)

    assert heavy.cost[0] > light.cost[0]
