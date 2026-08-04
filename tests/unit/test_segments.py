"""Per-segment difficulty and the elevation profile.

Two invariants carry the weight. The segment indices must tile the route's
geometry exactly — a client slices the polyline with them, so an off-by-one
draws a gap or a doubled point on every edge boundary. And the profile must
reconcile with the ascent and descent already reported, because it is summed
from the same per-edge figures and any disagreement means one of the two is
lying.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mtb_planner.cost.costvector import materialise
from mtb_planner.cost.profiles import RideProfile
from mtb_planner.graph.store import GraphStore
from mtb_planner.routing.kernels import warm_up
from mtb_planner.routing.result import build_route
from mtb_planner.routing.router import Router
from mtb_planner.routing.segments import elevation_profile, route_segments

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", autouse=True)
def _compiled() -> None:
    """Compile the kernels once."""
    warm_up()


def node_of(store: GraphStore, osm_id: int) -> int:
    """Graph index of a fixture node, by OSM id."""
    return int(np.flatnonzero(store.node_osm_id == osm_id)[0])


@pytest.fixture
def route(ardennes_graph: GraphStore, profile: RideProfile):
    """The longest route the fixture network offers."""
    costs = materialise(ardennes_graph, profile)
    result = Router(ardennes_graph, costs).shortest(
        node_of(ardennes_graph, 9), node_of(ardennes_graph, 15)
    )
    assert result.reached
    return build_route(ardennes_graph, node_of(ardennes_graph, 9), result.arcs, costs.time_s)


# --------------------------------------------------------------- indices


def test_there_is_one_segment_per_edge(route, ardennes_graph: GraphStore):
    assert len(route_segments(ardennes_graph, route)) == len(route.arcs)


def test_the_first_segment_starts_at_the_first_point(route, ardennes_graph: GraphStore):
    assert route_segments(ardennes_graph, route)[0].first == 0


def test_the_last_segment_ends_at_the_last_point(route, ardennes_graph: GraphStore):
    segments = route_segments(ardennes_graph, route)

    assert segments[-1].last == route.geometry.shape[0] - 1


def test_segments_tile_the_geometry_without_gap_or_overlap(route, ardennes_graph: GraphStore):
    # Consecutive edges share the junction between them, so each segment must
    # end exactly where the next begins.
    segments = route_segments(ardennes_graph, route)

    for earlier, later in itertools.pairwise(segments):
        assert earlier.last == later.first


def test_every_segment_covers_at_least_one_span(route, ardennes_graph: GraphStore):
    for segment in route_segments(ardennes_graph, route):
        assert segment.last > segment.first


def test_a_segment_slice_is_exactly_that_edges_geometry(route, ardennes_graph: GraphStore):
    # The property a client depends on: slicing the polyline by these indices
    # yields that edge and nothing else.
    segments = route_segments(ardennes_graph, route)

    for position, segment in enumerate(segments):
        expected = ardennes_graph.arc_geometry(route.arcs[position])
        actual = route.geometry[segment.first : segment.last + 1]
        np.testing.assert_allclose(actual, expected)


def test_an_empty_route_has_no_segments(ardennes_graph: GraphStore):
    empty = build_route(ardennes_graph, 0, ())

    assert route_segments(ardennes_graph, empty) == ()


# --------------------------------------------------------------- distances


def test_the_first_segment_begins_at_zero(route, ardennes_graph: GraphStore):
    assert route_segments(ardennes_graph, route)[0].at_m == 0.0


def test_distances_advance_by_each_segments_length(route, ardennes_graph: GraphStore):
    segments = route_segments(ardennes_graph, route)

    for earlier, later in itertools.pairwise(segments):
        assert later.at_m == pytest.approx(earlier.at_m + earlier.length_m)


def test_the_segments_account_for_the_whole_route(route, ardennes_graph: GraphStore):
    segments = route_segments(ardennes_graph, route)
    covered = segments[-1].at_m + segments[-1].length_m

    assert covered == pytest.approx(route.stats.distance_m, rel=1e-9)


# --------------------------------------------------------------- difficulty


def test_every_segment_reports_a_difficulty_on_the_scale(route, ardennes_graph: GraphStore):
    for segment in route_segments(ardennes_graph, route):
        assert 0.0 <= segment.tech <= 6.0


def test_difficulty_is_taken_in_the_direction_ridden(route, ardennes_graph: GraphStore):
    # A rooty descent is a different proposition climbed, and the artifact
    # records both directions. Conflating them is the classic mistake.
    features = ardennes_graph.features

    for segment, arc in zip(route_segments(ardennes_graph, route), route.arcs, strict=True):
        edge = int(ardennes_graph.arc_edge[arc])
        expected = (
            features.tech[edge] if ardennes_graph.arc_forward[arc] else features.tech_up[edge]
        )
        assert segment.tech == pytest.approx(float(expected))


def test_whether_difficulty_was_surveyed_is_carried(route, ardennes_graph: GraphStore):
    # The credibility field: at a few percent coverage, drawing inferred
    # difficulty like surveyed difficulty would claim more than is known.
    flags = {segment.surveyed for segment in route_segments(ardennes_graph, route)}

    assert flags
    assert flags <= {True, False}


# --------------------------------------------------------------- profile


def test_the_profile_has_a_point_per_segment_boundary(route, ardennes_graph: GraphStore):
    segments = route_segments(ardennes_graph, route)

    assert len(elevation_profile(segments)) == len(segments) + 1


def test_the_profile_starts_at_zero(route, ardennes_graph: GraphStore):
    # Relative, not altitude: there is no absolute height in the artifact to
    # anchor it to, so the start is the datum.
    points = elevation_profile(route_segments(ardennes_graph, route))

    assert points[0].at_m == 0.0
    assert points[0].height_m == 0.0


def test_the_profile_advances_monotonically_in_distance(route, ardennes_graph: GraphStore):
    points = elevation_profile(route_segments(ardennes_graph, route))

    for earlier, later in itertools.pairwise(points):
        assert later.at_m > earlier.at_m


def test_the_profile_ends_at_the_routes_length(route, ardennes_graph: GraphStore):
    points = elevation_profile(route_segments(ardennes_graph, route))

    assert points[-1].at_m == pytest.approx(route.stats.distance_m, rel=1e-9)


def test_the_net_height_change_reconciles_with_reported_ascent_and_descent(
    route, ardennes_graph: GraphStore
):
    # The honesty check. The chart and the summary figures come from the same
    # per-edge numbers, so they must agree; if they drift, one of them is wrong.
    points = elevation_profile(route_segments(ardennes_graph, route))

    assert points[-1].height_m == pytest.approx(
        route.stats.ascent_m - route.stats.descent_m, abs=1e-6
    )


def test_the_climbing_in_the_profile_matches_the_reported_ascent(route, ardennes_graph: GraphStore):
    points = elevation_profile(route_segments(ardennes_graph, route))
    climbed = sum(
        max(0.0, later.height_m - earlier.height_m) for earlier, later in itertools.pairwise(points)
    )

    assert climbed == pytest.approx(route.stats.ascent_m, abs=1e-6)


def test_an_empty_route_has_no_profile():
    assert elevation_profile(()) == ()
