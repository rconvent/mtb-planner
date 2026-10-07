"""Splitting and assembling the routable CSR graph.

Counts are derived by hand from the fixture layout documented in
`tests/fixtures/networks.py`, not copied from a previous run.

Structural invariants are asserted against the *uncontracted* graph, where
every junction is still a node and the arithmetic is checkable by eye.
Contraction has its own tests in `test_simplify.py`; what matters here is that
the assembled arrays stay consistent either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from planner_core.graph.builder import build_graph
from planner_core.graph.store import GraphStore
from planner_core.ingest.pbf_reader import IngestResult, read_extracts

pytestmark = pytest.mark.unit


def node_index_of(store: GraphStore, osm_id: int) -> int:
    """Compact index of an OSM node id."""
    matches = np.flatnonzero(store.node_osm_id == osm_id)
    assert matches.size == 1, f"OSM node {osm_id} is not in the graph"
    return int(matches[0])


# --------------------------------------------------------------- splitting


def test_junctions_become_nodes_and_interior_points_do_not(
    ardennes_graph_uncontracted: GraphStore,
):
    # Nine grid corners, two on the isolated fragment, one at the top of the
    # steps. Node 13 is interior to a single way, so it stays geometry.
    assert ardennes_graph_uncontracted.n_nodes == 12
    assert 13 not in set(ardennes_graph_uncontracted.node_osm_id.tolist())


def test_ways_are_split_at_every_junction(ardennes_graph_uncontracted: GraphStore):
    # Six three-node grid ways split into two edges each, plus the fragment,
    # the descent, and the steps.
    assert ardennes_graph_uncontracted.n_edges == 15


def test_every_edge_yields_a_forward_and_a_backward_arc(ardennes_graph: GraphStore):
    assert ardennes_graph.n_arcs == 2 * ardennes_graph.n_edges


def test_only_the_corroborated_oneway_blocks_an_arc(ardennes_graph: GraphStore):
    # Exactly one arc is disallowed: the reverse of the one-way descent.
    assert ardennes_graph.n_allowed_arcs == ardennes_graph.n_arcs - 1

    blocked = np.flatnonzero(ardennes_graph.arc_allowed == 0)
    blocked_edge = int(ardennes_graph.arc_edge[blocked[0]])
    assert int(ardennes_graph.edge_way_id[blocked_edge]) == 202


# --------------------------------------------------------------- arcs


def test_reverse_arcs_are_an_involution(ardennes_graph: GraphStore):
    # U-turn prohibition depends on this holding exactly.
    rev = ardennes_graph.arc_rev
    np.testing.assert_array_equal(rev[rev], np.arange(ardennes_graph.n_arcs))


def test_an_arc_and_its_twin_share_an_edge_and_swap_endpoints(ardennes_graph: GraphStore):
    store = ardennes_graph
    for arc in range(store.n_arcs):
        twin = int(store.arc_rev[arc])
        assert int(store.arc_edge[arc]) == int(store.arc_edge[twin])
        assert store.arc_tail(arc) == int(store.arc_head[twin])
        assert int(store.arc_head[arc]) == store.arc_tail(twin)


def test_csr_offsets_are_consistent(ardennes_graph: GraphStore):
    store = ardennes_graph
    assert store.indptr.shape[0] == store.n_nodes + 1
    assert int(store.indptr[0]) == 0
    assert int(store.indptr[-1]) == store.n_arcs
    assert np.all(np.diff(store.indptr) >= 0)


def test_arcs_are_grouped_by_their_source_node(ardennes_graph: GraphStore):
    store = ardennes_graph
    for node in range(store.n_nodes):
        for arc in store.arcs_of(node):
            assert store.arc_tail(arc) == node


# --------------------------------------------------------------- geometry


def test_geometry_offsets_cover_the_whole_blob(ardennes_graph: GraphStore):
    store = ardennes_graph
    assert store.edge_geom_offset.shape[0] == store.n_edges + 1
    assert int(store.edge_geom_offset[0]) == 0
    assert int(store.edge_geom_offset[-1]) == store.geom.shape[0]
    assert np.all(np.diff(store.edge_geom_offset) >= 2)


def test_edge_geometry_starts_and_ends_at_its_endpoints(ardennes_graph: GraphStore):
    store = ardennes_graph
    for edge in range(store.n_edges):
        points = store.edge_geometry(edge)
        u, v = int(store.edge_u[edge]), int(store.edge_v[edge])
        assert points[0, 0] == pytest.approx(store.node_lat[u])
        assert points[-1, 0] == pytest.approx(store.node_lat[v])


def test_reversed_geometry_is_the_forward_geometry_backwards(ardennes_graph: GraphStore):
    forward = ardennes_graph.edge_geometry(0, forward=True)
    backward = ardennes_graph.edge_geometry(0, forward=False)

    np.testing.assert_array_equal(backward, forward[::-1])


def test_the_descent_keeps_its_interior_geometry(ardennes_graph: GraphStore):
    # Way 202 runs 9 -> 13 -> 3. Node 13 is not a junction, so the edge must
    # carry three points rather than being straightened to two.
    edge = int(np.flatnonzero(ardennes_graph.edge_way_id == 202)[0])

    assert ardennes_graph.edge_geometry(edge).shape[0] == 3


def test_every_edge_has_positive_length(ardennes_graph: GraphStore):
    assert np.all(ardennes_graph.edge_length_m > 0)


def test_grid_edge_lengths_match_the_fixture_spacing(
    ardennes_graph_uncontracted: GraphStore,
):
    # Grid steps are 0.001 degrees: about 111 m north-south and, at 50°N,
    # about 71 m east-west. Restricted to the grid ways, because the steps and
    # the descent are deliberately diagonal.
    store = ardennes_graph_uncontracted
    grid = np.isin(store.edge_way_id, [101, 102, 103, 104, 105, 106])
    lengths = store.edge_length_m[grid]

    assert lengths.min() == pytest.approx(71.5, abs=2.0)
    assert lengths.max() == pytest.approx(111.2, abs=2.0)


def test_total_length_is_preserved_by_contraction(
    ardennes_graph: GraphStore, ardennes_graph_uncontracted: GraphStore
):
    # Merging edges must move no geometry, only relabel it.
    assert ardennes_graph.total_length_m() == pytest.approx(
        ardennes_graph_uncontracted.total_length_m(), rel=1e-9
    )


# --------------------------------------------------------------- components


def test_the_isolated_fragment_is_kept_out_of_the_main_component(
    ardennes_graph_uncontracted: GraphStore,
):
    store = ardennes_graph_uncontracted
    # Nine grid corners plus the top of the steps, which is reachable both ways.
    assert store.giant_scc_size == 10

    island = node_index_of(store, 11)
    assert not store.in_giant_component(island)


def test_grid_nodes_are_all_in_the_main_component(ardennes_graph_uncontracted: GraphStore):
    for osm_id in range(1, 10):
        assert ardennes_graph_uncontracted.in_giant_component(
            node_index_of(ardennes_graph_uncontracted, osm_id)
        )


def test_bbox_covers_every_node(ardennes_graph: GraphStore):
    store = ardennes_graph
    box = store.bbox()

    assert box.min_lat == pytest.approx(float(store.node_lat.min()))
    assert box.max_lon == pytest.approx(float(store.node_lon.max()))


# --------------------------------------------------------------- determinism


def test_building_the_same_extract_twice_gives_identical_arrays(ardennes_pbf: Path):
    # Arc ordering feeds every search, so a non-deterministic layout would make
    # routes wobble between runs for no reason the user could see.
    first = build_graph(read_extracts([ardennes_pbf]))
    second = build_graph(read_extracts([ardennes_pbf]))

    np.testing.assert_array_equal(first.arc_head, second.arc_head)
    np.testing.assert_array_equal(first.arc_edge, second.arc_edge)
    np.testing.assert_array_equal(first.arc_rev, second.arc_rev)
    np.testing.assert_array_equal(first.indptr, second.indptr)
    np.testing.assert_allclose(first.edge_length_m, second.edge_length_m)


def test_excluded_ways_leave_no_trace_in_the_graph(ardennes_ingest: IngestResult):
    store = build_graph(ardennes_ingest)
    way_ids = set(store.edge_way_id.tolist())

    assert 200 not in way_ids
    assert 203 not in way_ids
