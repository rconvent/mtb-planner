"""The spatial prefilter.

Everything the loop generator does happens inside this ball, so two properties
carry the weight: a search confined to it must be unable to step outside, and
restricting to it must not change what a route costs. The second is what makes
the prefilter a speedup rather than a different algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtb_planner.cost.costvector import materialise, max_speed_mps
from mtb_planner.cost.profiles import RideProfile
from mtb_planner.cost.speed import Rider
from mtb_planner.graph.store import GraphStore
from mtb_planner.routing.ball import (
    MIN_RADIUS_M,
    arc_tails,
    ball_around,
    edges_inside,
    loop_radius_m,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------- radius


def test_a_longer_budget_reaches_further():
    assert loop_radius_m(7200.0, 9.0) > loop_radius_m(3600.0, 9.0)


def test_the_radius_is_half_the_budget_at_full_speed():
    # Half, because a loop has to come back: the outbound leg gets half the
    # time, and the bound assumes a straight line at the fastest speed.
    assert loop_radius_m(7200.0, 9.0) == pytest.approx(0.5 * 7200.0 * 9.0)


def test_even_a_tiny_budget_gets_somewhere_to_go():
    assert loop_radius_m(60.0, 9.0) == MIN_RADIUS_M


def test_the_top_speed_is_a_property_of_the_rider_not_the_graph():
    # The ball has to be sized before a cost vector exists, because the cost
    # vector is built over the ball.
    assert 1.0 < max_speed_mps(Rider()) < 30.0


def test_a_stronger_rider_is_never_slower():
    assert max_speed_mps(Rider(power_w=280.0)) >= max_speed_mps(Rider(power_w=130.0))


# --------------------------------------------------------------- extraction


def test_arc_tails_agree_with_the_stores_own_lookup(ardennes_graph: GraphStore):
    tails = arc_tails(ardennes_graph)

    assert tails.shape[0] == ardennes_graph.n_arcs
    for arc in range(ardennes_graph.n_arcs):
        assert int(tails[arc]) == ardennes_graph.arc_tail(arc)


def test_a_generous_ball_holds_the_whole_fixture_network(ardennes_graph: GraphStore):
    ball = ball_around(ardennes_graph, 0, 100_000.0)

    assert ball.n_nodes == ardennes_graph.n_nodes
    assert ball.n_arcs == ardennes_graph.n_arcs


def test_a_tight_ball_excludes_the_distant_fragment(ardennes_graph: GraphStore):
    # The isolated fragment sits 11 km north-east of the grid.
    grid_node = int(np.flatnonzero(ardennes_graph.node_osm_id == 1)[0])
    island = int(np.flatnonzero(ardennes_graph.node_osm_id == 11)[0])

    ball = ball_around(ardennes_graph, grid_node, 1_000.0)

    assert ball.contains_node(grid_node)
    assert not ball.contains_node(island)


def test_the_centre_is_always_inside_its_own_ball(ardennes_graph: GraphStore):
    for node in range(ardennes_graph.n_nodes):
        assert ball_around(ardennes_graph, node, MIN_RADIUS_M).contains_node(node)


def test_an_arc_is_inside_only_when_both_its_ends_are(ardennes_graph: GraphStore):
    # This is what stops a search leaving the ball: there is no arc out of it.
    grid_node = int(np.flatnonzero(ardennes_graph.node_osm_id == 1)[0])
    ball = ball_around(ardennes_graph, grid_node, 150.0)
    tails = arc_tails(ardennes_graph)

    for arc in np.flatnonzero(ball.arc_inside):
        assert ball.contains_node(int(tails[arc]))
        assert ball.contains_node(int(ardennes_graph.arc_head[arc]))


def test_a_zero_radius_ball_has_no_arcs_to_ride(ardennes_graph: GraphStore):
    ball = ball_around(ardennes_graph, 0, 0.0)

    assert ball.n_nodes == 1
    assert ball.n_arcs == 0


def test_the_edge_selection_covers_every_arc_inside(ardennes_graph: GraphStore):
    ball = ball_around(ardennes_graph, 0, 200.0)

    edges = edges_inside(ardennes_graph, ball)

    for arc in np.flatnonzero(ball.arc_inside):
        assert edges[int(ardennes_graph.arc_edge[arc])]


# --------------------------------------------------------------- cost vector


def test_restricting_the_cost_vector_leaves_costs_unchanged(
    ardennes_graph: GraphStore, profile: RideProfile
):
    # The prefilter must be a speedup, not a different cost model. Anything
    # else would make a loop's figures disagree with a point-to-point route
    # over the same ground.
    full = materialise(ardennes_graph, profile)
    ball = ball_around(ardennes_graph, 0, 100_000.0)
    restricted = materialise(ardennes_graph, profile, arc_mask=ball.arc_inside)

    np.testing.assert_allclose(restricted.cost, full.cost)
    np.testing.assert_allclose(restricted.time_s, full.time_s)
    np.testing.assert_allclose(restricted.distance_m, full.distance_m)


def test_arcs_outside_the_ball_are_not_traversable(
    ardennes_graph: GraphStore, profile: RideProfile
):
    grid_node = int(np.flatnonzero(ardennes_graph.node_osm_id == 1)[0])
    ball = ball_around(ardennes_graph, grid_node, 150.0)

    costs = materialise(ardennes_graph, profile, arc_mask=ball.arc_inside)

    assert not costs.allowed[~ball.arc_inside].any()


def test_a_restricted_vector_keeps_the_arcs_inside_usable(
    ardennes_graph: GraphStore, profile: RideProfile
):
    full = materialise(ardennes_graph, profile)
    ball = ball_around(ardennes_graph, 0, 100_000.0)

    restricted = materialise(ardennes_graph, profile, arc_mask=ball.arc_inside)

    np.testing.assert_array_equal(restricted.allowed, full.allowed)


def test_the_heuristic_scale_survives_restriction(ardennes_graph: GraphStore, profile: RideProfile):
    # A* admissibility depends on this, and it is derived from the arrays the
    # restriction rewrites.
    ball = ball_around(ardennes_graph, 0, 100_000.0)
    restricted = materialise(ardennes_graph, profile, arc_mask=ball.arc_inside)

    assert restricted.min_seconds_per_metre == pytest.approx(
        materialise(ardennes_graph, profile).min_seconds_per_metre
    )
