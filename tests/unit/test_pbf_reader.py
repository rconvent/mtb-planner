"""Reading extracts into ways, nodes, and route relations."""

from __future__ import annotations

from pathlib import Path

import pytest

from planner_core.core.errors import PbfParseError
from planner_core.core.geo import BBox
from planner_core.ingest.pbf_reader import IngestResult, read_extracts
from planner_core.sport import Access, Direction
from tests.fixtures.networks import border_pair, empty_of_ways, single_track

pytestmark = pytest.mark.unit


def test_rideable_ways_are_kept(ardennes_ingest: IngestResult):
    # The six grid ways, the isolated fragment, the descent, and the steps.
    assert set(ardennes_ingest.ways.ids.tolist()) == {101, 102, 103, 104, 105, 106, 201, 202, 204}


def test_way_ids_are_sorted_so_lookups_can_binary_search(ardennes_ingest: IngestResult):
    ids = ardennes_ingest.ways.ids
    assert list(ids) == sorted(ids)


def test_node_ids_are_sorted_and_unique(ardennes_ingest: IngestResult):
    ids = ardennes_ingest.nodes.ids
    assert list(ids) == sorted(set(ids.tolist()))


def test_identical_tags_are_interned_rather_than_duplicated(ardennes_ingest: IngestResult):
    # The six grid ways carry the same tags; holding six copies of that object
    # is what makes a country-scale build run out of memory.
    grid = [ardennes_ingest.ways.row(way_id).tags for way_id in (101, 102, 103, 104, 105, 106)]

    assert all(tags is grid[0] for tags in grid)


@pytest.mark.parametrize(
    ("way_id", "why"),
    [
        (200, "a private driveway"),
        (203, "tagged bicycle=no"),
    ],
)
def test_legally_closed_ways_never_enter_the_graph(ardennes_ingest: IngestResult, way_id, why):
    assert way_id not in ardennes_ingest.ways, f"way {way_id} is {why} and must be excluded"


def test_a_bicycle_route_relation_is_recorded_against_its_way(ardennes_ingest: IngestResult):
    assert 901 in ardennes_ingest.way_to_routes[102]


def test_only_nodes_referenced_by_kept_ways_are_loaded(ardennes_ingest: IngestResult):
    # Nodes 10 and 14 belong solely to the excluded ways, so reading their
    # coordinates would be wasted memory at country scale.
    assert 10 not in ardennes_ingest.nodes
    assert 14 not in ardennes_ingest.nodes
    assert ardennes_ingest.n_nodes == 13


def test_steps_are_kept_as_a_pushing_section(ardennes_ingest: IngestResult):
    assert ardennes_ingest.ways.row(204).access is Access.WALK


def test_a_corroborated_oneway_descent_stays_directional(ardennes_ingest: IngestResult):
    descent = ardennes_ingest.ways.row(202)

    assert descent.direction is Direction.FORWARD
    assert descent.oneway_corroborated is True


def test_junction_reference_counts_identify_crossings(ardennes_ingest: IngestResult):
    # Node 5 is the centre of the grid: one row way and one column way.
    assert ardennes_ingest.nodes.ref_count_of(5) == 2
    # Node 13 is interior to a single way, so it is pure geometry.
    assert ardennes_ingest.nodes.ref_count_of(13) == 1


def test_node_references_round_trip_through_the_columnar_layout(ardennes_ingest: IngestResult):
    # Way 202 runs 9 -> 13 -> 3; the offsets must recover exactly that run.
    assert ardennes_ingest.ways.row(202).node_refs.tolist() == [9, 13, 3]


def test_mtb_route_relations_are_collected(ardennes_ingest: IngestResult):
    routes = ardennes_ingest.signed_routes()

    assert len(routes) == 1
    relation = next(iter(routes.values()))
    assert relation.name == "Rode lus"
    assert relation.network == "lcn"
    assert relation.way_ids == frozenset({101, 106})


def test_bicycle_routes_are_kept_separately_from_mtb_routes(ardennes_ingest: IngestResult):
    # Knooppunten networks are connectors, not riding, and must not be
    # mistaken for signed mountain-bike loops.
    assert 901 in ardennes_ingest.route_relations
    assert ardennes_ingest.route_relations[901].route_type == "bicycle"
    assert 901 not in ardennes_ingest.signed_routes()


def test_ways_are_mapped_back_to_the_routes_that_contain_them(ardennes_ingest: IngestResult):
    assert 900 in ardennes_ingest.way_to_routes[101]
    assert 900 in ardennes_ingest.way_to_routes[106]


def test_a_single_way_extract_reads(tmp_path: Path):
    result = read_extracts([single_track(tmp_path)])

    assert result.n_ways == 1
    assert result.n_nodes == 3


# --------------------------------------------------------------- merging


def test_merging_extracts_keeps_the_longer_copy_of_a_shared_way(tmp_path: Path):
    # Border-clipped copies are truncated. Keeping the short one would build a
    # routing wall exactly along the frontier.
    west, east = border_pair(tmp_path)

    result = read_extracts([west, east])

    assert result.ways.row(500).node_refs.shape[0] == 4


def test_merge_order_does_not_change_the_result(tmp_path: Path):
    west, east = border_pair(tmp_path)

    forwards = read_extracts([west, east])
    backwards = read_extracts([east, west])

    assert forwards.ways.row(500).node_refs.tolist() == backwards.ways.row(500).node_refs.tolist()


def test_relation_members_are_unioned_across_extracts(tmp_path: Path):
    # A relation clipped at a border loses members in one file and has them in
    # the other; taking either alone would under-report the route.
    west, east = border_pair(tmp_path)

    result = read_extracts([west, east])

    assert result.route_relations[950].way_ids == frozenset({500, 501})


# --------------------------------------------------------------- clipping


def test_a_bounding_box_drops_ways_that_lie_entirely_outside(ardennes_pbf: Path):
    # The grid sits near 50.10/5.10; the isolated fragment is 11 km north.
    grid_only = BBox(min_lon=5.09, min_lat=50.09, max_lon=5.11, max_lat=50.11)

    result = read_extracts([ardennes_pbf], bbox=grid_only)

    assert 201 not in result.ways
    assert result.dropped_outside_bbox_ways == 1


def test_a_way_with_one_node_inside_the_box_is_kept(ardennes_pbf: Path):
    # Clipping on the first out-of-box node would sever every way crossing the
    # boundary, which is precisely where the good riding is.
    clipped = BBox(min_lon=5.0995, min_lat=50.0995, max_lon=5.1005, max_lat=50.1005)

    result = read_extracts([ardennes_pbf], bbox=clipped)

    # Way 103 runs east from node 7, which is the only node inside the box.
    assert 103 in result.ways


# --------------------------------------------------------------- failures


def test_a_missing_extract_is_reported_clearly(tmp_path: Path):
    with pytest.raises(PbfParseError, match="not found"):
        read_extracts([tmp_path / "nope.osm.pbf"])


def test_no_paths_at_all_is_an_error():
    with pytest.raises(PbfParseError, match="No extract paths"):
        read_extracts([])


def test_an_extract_with_nothing_rideable_fails_loudly(tmp_path: Path):
    with pytest.raises(PbfParseError, match="No mtb ways"):
        read_extracts([empty_of_ways(tmp_path)])


def test_a_bounding_box_that_excludes_everything_fails_loudly(ardennes_pbf: Path):
    # Silently returning an empty graph would look like a build succeeding.
    with pytest.raises(PbfParseError, match="check the bounding box"):
        read_extracts([ardennes_pbf], bbox=BBox(2.0, 48.0, 2.1, 48.1))
