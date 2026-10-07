"""Contracting chains that pass straight through a node.

The fixture grid contains exactly one contractible node, which makes the
expected outcome checkable by hand: node 7 is the south-west corner, where the
bottom row way and the left column way meet with identical tags and nothing
else touching it.
"""

from __future__ import annotations

import numpy as np
import pytest

from planner_core.graph.segments import Segment, split_ways
from planner_core.graph.simplify import contract
from planner_core.graph.store import GraphStore
from planner_core.ingest.pbf_reader import IngestResult
from planner_core.sport import Direction

pytestmark = pytest.mark.unit


@pytest.fixture
def segments(ardennes_ingest: IngestResult) -> list[Segment]:
    """The fixture network split at its junctions, before contraction."""
    return split_ways(ardennes_ingest.nodes, ardennes_ingest.ways, ardennes_ingest.way_to_routes)


def test_splitting_produces_one_segment_per_junction_gap(segments: list[Segment]):
    # Six three-node grid ways give two each; the fragment, descent and steps
    # give one each.
    assert len(segments) == 15


def test_contraction_removes_the_single_mergeable_node(
    segments: list[Segment], ardennes_ingest: IngestResult
):
    merged = contract(segments, ardennes_ingest.nodes)

    assert len(merged) == len(segments) - 1


def test_the_merged_edge_spans_both_original_endpoints(
    segments: list[Segment], ardennes_ingest: IngestResult
):
    # Node 7 joins the bottom row (7 -> 8) and the left column (4 -> 7), so the
    # merged edge must run between nodes 4 and 8.
    merged = contract(segments, ardennes_ingest.nodes)

    spans = {frozenset((s.u, s.v)) for s in merged}
    assert frozenset((4, 8)) in spans
    assert frozenset((4, 7)) not in spans
    assert frozenset((7, 8)) not in spans


def test_the_merged_edge_keeps_the_vanished_node_as_geometry(
    segments: list[Segment], ardennes_ingest: IngestResult
):
    # Contraction must relabel, never straighten: node 7 is still a corner of
    # the trail even though it is no longer a junction.
    merged = contract(segments, ardennes_ingest.nodes)

    joined = next(s for s in merged if frozenset((s.u, s.v)) == frozenset((4, 8)))
    assert 7 in joined.node_ids.tolist()
    assert joined.node_ids.shape[0] == 3


def test_contraction_preserves_every_node_reference(
    segments: list[Segment], ardennes_ingest: IngestResult
):
    # No geometry may be lost, so every node that appeared before must still
    # appear somewhere afterwards.
    before = {int(n) for s in segments for n in s.node_ids}
    after = {int(n) for s in contract(segments, ardennes_ingest.nodes) for n in s.node_ids}

    assert before == after


def test_contraction_is_idempotent(segments: list[Segment], ardennes_ingest: IngestResult):
    once = contract(segments, ardennes_ingest.nodes)
    twice = contract(once, ardennes_ingest.nodes)

    assert len(twice) == len(once)


def test_an_empty_network_contracts_to_nothing(ardennes_ingest: IngestResult):
    assert contract([], ardennes_ingest.nodes) == []


def test_the_oneway_descent_is_never_merged_into_a_two_way_edge(
    segments: list[Segment], ardennes_ingest: IngestResult
):
    # Merging a directional edge with a bidirectional one would silently open
    # a descent to uphill traffic.
    merged = contract(segments, ardennes_ingest.nodes)

    descent = [s for s in merged if s.way_id == 202]
    assert len(descent) == 1

    # Storage orientation is not canonical, so assert the property that is:
    # exactly one direction is rideable, and it runs downhill from node 9.
    edge = descent[0]
    assert edge.direction != Direction.BOTH
    downhill = edge if edge.direction == Direction.FORWARD else edge.reversed()
    assert (downhill.u, downhill.v) == (9, 3)


def test_reversing_a_segment_flips_its_direction(segments: list[Segment]):
    descent = next(s for s in segments if s.way_id == 202)

    flipped = descent.reversed()

    assert descent.direction == Direction.FORWARD
    assert flipped.direction == Direction.BACKWARD
    assert flipped.u == descent.v
    assert flipped.v == descent.u
    np.testing.assert_array_equal(flipped.node_ids, descent.node_ids[::-1])


def test_reversing_twice_is_the_identity(segments: list[Segment]):
    original = segments[0]

    round_trip = original.reversed().reversed()

    assert round_trip.direction == original.direction
    np.testing.assert_array_equal(round_trip.node_ids, original.node_ids)


def test_a_two_way_segment_reverses_without_changing_direction(segments: list[Segment]):
    two_way = next(s for s in segments if s.direction == Direction.BOTH)

    assert two_way.reversed().direction == Direction.BOTH


# --------------------------------------------------------------- integration


def test_contraction_leaves_the_graph_routable(
    ardennes_graph: GraphStore, ardennes_graph_uncontracted: GraphStore
):
    # Fewer nodes and edges, same reachable network.
    assert ardennes_graph.n_nodes < ardennes_graph_uncontracted.n_nodes
    assert ardennes_graph.n_edges < ardennes_graph_uncontracted.n_edges
    assert ardennes_graph.giant_scc_size > 0


def test_contraction_preserves_the_geometry_point_count(
    ardennes_graph: GraphStore, ardennes_graph_uncontracted: GraphStore
):
    # Each merge drops the duplicated junction point, and nothing else.
    dropped_edges = ardennes_graph_uncontracted.n_edges - ardennes_graph.n_edges

    assert ardennes_graph.geom.shape[0] == ardennes_graph_uncontracted.geom.shape[0] - dropped_edges
