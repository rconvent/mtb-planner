"""Spreading surveyed difficulty across a trail.

Groups are built from three signals — shared name, shared route relation, and
corridor adjacency between trail edges of matching surface. The last is what
carries the weight in practice: measured against Luxembourg, only 111 of 4,590
surveyed segments also carry a name, so nominal grouping alone reaches almost
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from planner_core.features.enums import SurfaceClass, TechConfidence, WayClass
from planner_core.features.model import EdgeFeatures
from planner_core.features.propagate import (
    GROUP_WEIGHT,
    confidence_coverage,
    propagate_difficulty,
)
from planner_core.graph.segments import Segment
from planner_core.ingest.tags import WayTags
from planner_core.sport import Access, Direction

pytestmark = pytest.mark.unit


def segment(
    u: int,
    v: int,
    name: str | None = None,
    routes: frozenset[int] = frozenset(),
) -> Segment:
    """A minimal segment carrying only what grouping reads."""
    return Segment(
        node_ids=np.array([u, v], dtype=np.int64),
        way_id=u * 1000 + v,
        tags=WayTags(highway="path", name=name),
        access=int(Access.YES),
        direction=int(Direction.BOTH),
        routes=routes,
    )


def features_for(
    tech: list[float],
    confidence: list[TechConfidence],
    surface: SurfaceClass = SurfaceClass.GROUND,
    way_class: WayClass = WayClass.SINGLETRACK,
) -> EdgeFeatures:
    """Feature arrays with only the columns propagation reads set meaningfully."""
    count = len(tech)
    return EdgeFeatures(
        tech=np.array(tech, dtype=np.float32),
        tech_up=np.zeros(count, dtype=np.float32),
        tech_conf=np.array([int(c) for c in confidence], dtype=np.uint8),
        singletrack=np.zeros(count, dtype=np.float32),
        rollability=np.zeros(count, dtype=np.float32),
        flow=np.zeros(count, dtype=np.float32),
        traffic=np.zeros(count, dtype=np.float32),
        surface_class=np.full(count, int(surface), dtype=np.uint8),
        way_class=np.full(count, int(way_class), dtype=np.uint8),
        n_signed_routes=np.zeros(count, dtype=np.uint8),
        route_flags=np.zeros(count, dtype=np.uint8),
    )


EQUAL_LENGTHS = np.array([100.0, 100.0], dtype=np.float64)


def test_a_surveyed_neighbour_lifts_an_estimate_to_proxy():
    segments = [segment(1, 2, name="Rode lus"), segment(2, 3, name="Rode lus")]
    features = features_for([4.0, 1.0], [TechConfidence.SURVEYED, TechConfidence.ESTIMATED])

    result = propagate_difficulty(features, segments, EQUAL_LENGTHS)

    assert result.tech_conf[1] == int(TechConfidence.PROXY)


def test_the_propagated_value_blends_group_and_own_estimate():
    # The edge's own estimate still carries real information: surface and
    # width were measured on that edge.
    segments = [segment(1, 2, name="Rode lus"), segment(2, 3, name="Rode lus")]
    features = features_for([4.0, 1.0], [TechConfidence.SURVEYED, TechConfidence.ESTIMATED])

    result = propagate_difficulty(features, segments, EQUAL_LENGTHS)

    expected = GROUP_WEIGHT * 4.0 + (1.0 - GROUP_WEIGHT) * 1.0
    assert result.tech[1] == pytest.approx(expected, rel=1e-5)


def test_a_surveyed_edge_is_never_overwritten():
    segments = [segment(1, 2, name="Rode lus"), segment(2, 3, name="Rode lus")]
    features = features_for([4.0, 1.0], [TechConfidence.SURVEYED, TechConfidence.ESTIMATED])

    result = propagate_difficulty(features, segments, EQUAL_LENGTHS)

    assert result.tech[0] == pytest.approx(4.0)
    assert result.tech_conf[0] == int(TechConfidence.SURVEYED)


def test_a_group_with_too_little_survey_is_left_alone():
    # One surveyed metre in a kilometre says nothing about the kilometre.
    segments = [segment(1, 2, name="Long trail"), segment(2, 3, name="Long trail")]
    features = features_for([4.0, 1.0], [TechConfidence.SURVEYED, TechConfidence.ESTIMATED])
    lopsided = np.array([1.0, 999.0], dtype=np.float64)

    result = propagate_difficulty(features, segments, lopsided)

    assert result.tech_conf[1] == int(TechConfidence.ESTIMATED)
    assert result.tech[1] == pytest.approx(1.0)


def test_nothing_surveyed_means_nothing_changes():
    segments = [segment(1, 2, name="Rode lus"), segment(2, 3, name="Rode lus")]
    features = features_for([1.0, 2.0], [TechConfidence.ESTIMATED, TechConfidence.ESTIMATED])

    result = propagate_difficulty(features, segments, EQUAL_LENGTHS)

    np.testing.assert_array_equal(result.tech, features.tech)
    np.testing.assert_array_equal(result.tech_conf, features.tech_conf)


def test_unrelated_trails_do_not_share_knowledge():
    # Different names, no shared node: nothing links them.
    segments = [segment(1, 2, name="Red"), segment(8, 9, name="Blue")]
    features = features_for([4.0, 1.0], [TechConfidence.SURVEYED, TechConfidence.ESTIMATED])

    result = propagate_difficulty(features, segments, EQUAL_LENGTHS)

    assert result.tech_conf[1] == int(TechConfidence.ESTIMATED)


def test_a_shared_route_relation_groups_unnamed_trails():
    segments = [segment(1, 2, routes=frozenset({900})), segment(8, 9, routes=frozenset({900}))]
    features = features_for([4.0, 1.0], [TechConfidence.SURVEYED, TechConfidence.ESTIMATED])

    result = propagate_difficulty(features, segments, EQUAL_LENGTHS)

    assert result.tech_conf[1] == int(TechConfidence.PROXY)


# --------------------------------------------------------------- corridors


def test_touching_trails_of_the_same_surface_form_a_corridor():
    # Neither is named nor routed; adjacency is the only link, and in real
    # data it is the one that reaches surveyed ways.
    segments = [segment(1, 2), segment(2, 3)]
    features = features_for([4.0, 1.0], [TechConfidence.SURVEYED, TechConfidence.ESTIMATED])

    result = propagate_difficulty(features, segments, EQUAL_LENGTHS)

    assert result.tech_conf[1] == int(TechConfidence.PROXY)


def test_a_corridor_does_not_cross_a_change_of_surface():
    segments = [segment(1, 2), segment(2, 3)]
    features = features_for([4.0, 1.0], [TechConfidence.SURVEYED, TechConfidence.ESTIMATED])
    # Second edge is gravel where the first is dirt.
    features.surface_class[1] = int(SurfaceClass.GRAVEL)

    result = propagate_difficulty(features, segments, EQUAL_LENGTHS)

    assert result.tech_conf[1] == int(TechConfidence.ESTIMATED)


def test_a_corridor_does_not_run_out_onto_a_fire_road():
    # The failure this guard exists for: a surveyed S4 descent bleeding into
    # the forestry road it lands on.
    segments = [segment(1, 2), segment(2, 3)]
    features = features_for([4.0, 1.0], [TechConfidence.SURVEYED, TechConfidence.ESTIMATED])
    features.way_class[1] = int(WayClass.FIREROAD)

    result = propagate_difficulty(features, segments, EQUAL_LENGTHS)

    assert result.tech_conf[1] == int(TechConfidence.ESTIMATED)


def test_edges_that_never_meet_are_not_a_corridor():
    segments = [segment(1, 2), segment(3, 4)]
    features = features_for([4.0, 1.0], [TechConfidence.SURVEYED, TechConfidence.ESTIMATED])

    result = propagate_difficulty(features, segments, EQUAL_LENGTHS)

    assert result.tech_conf[1] == int(TechConfidence.ESTIMATED)


# --------------------------------------------------------------- coverage


def test_coverage_reports_the_share_of_length_at_each_level():
    features = features_for([4.0, 1.0], [TechConfidence.SURVEYED, TechConfidence.ESTIMATED])
    lengths = np.array([300.0, 100.0], dtype=np.float64)

    coverage = confidence_coverage(features, lengths)

    assert coverage["surveyed"] == pytest.approx(0.75)
    assert coverage["estimated"] == pytest.approx(0.25)
    assert sum(coverage.values()) == pytest.approx(1.0)


def test_coverage_is_measured_in_length_not_edge_count():
    # A thousand short urban links must not flatter the figure.
    features = features_for(
        [4.0, 1.0, 1.0],
        [TechConfidence.SURVEYED, TechConfidence.DEFAULT, TechConfidence.DEFAULT],
    )
    lengths = np.array([900.0, 50.0, 50.0], dtype=np.float64)

    coverage = confidence_coverage(features, lengths)

    assert coverage["surveyed"] == pytest.approx(0.9)


def test_coverage_of_an_empty_network_is_zero_everywhere():
    features = features_for([], [])

    coverage = confidence_coverage(features, np.array([], dtype=np.float64))

    assert set(coverage) == {"default", "estimated", "proxy", "surveyed"}
    assert all(share == 0.0 for share in coverage.values())
