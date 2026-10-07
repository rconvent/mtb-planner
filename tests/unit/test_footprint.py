"""Which ground a build has to have elevation for.

An extract carries more coordinates than its country rides on: ways the clip
box discarded keep theirs, and Geofabrik ships ferry lines whole, so the
Belgian extract holds nodes in the Channel and the middle of the North Sea.
Measuring those in asks for GLO-30 tiles over open water, which do not exist,
and the build then refuses to start over ground nobody rides.
"""

from __future__ import annotations

import numpy as np
import pytest

from planner_core.core.errors import BuildError
from planner_core.core.units import COORD_SCALE
from planner_core.elevation.dem_catalog import tiles_for_bbox
from planner_core.elevation.footprint import elevation_footprint
from planner_core.ingest.tables import NodeTable, WayTable
from planner_core.ingest.tags import WayTags
from planner_core.sport import Access, Direction

pytestmark = pytest.mark.unit


def _tables(
    points: list[tuple[float, float]],
    ways: list[tuple[list[int], WayTags]],
) -> tuple[WayTable, NodeTable]:
    """Tables from `(lat, lon)` coordinates and `(node indices, tags)` ways.

    Node ids are positions in `points`, which keeps them sorted and unique as
    `NodeTable` requires.
    """
    lat, lon = zip(*points, strict=True)
    nodes = NodeTable(
        ids=np.arange(len(points), dtype=np.int64),
        lat_e7=np.asarray([round(value * COORD_SCALE) for value in lat], dtype=np.int32),
        lon_e7=np.asarray([round(value * COORD_SCALE) for value in lon], dtype=np.int32),
        ref_count=np.zeros(len(points), dtype=np.int32),
        barrier=np.zeros(len(points), dtype=np.bool_),
    )

    refs = [ref for way_refs, _ in ways for ref in way_refs]
    lengths = [len(way_refs) for way_refs, _ in ways]
    way_table = WayTable(
        ids=np.arange(len(ways), dtype=np.int64),
        node_offset=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        node_refs=np.asarray(refs, dtype=np.int64),
        tags=tuple(tags for _, tags in ways),
        access=np.full(len(ways), Access.YES, dtype=np.uint8),
        direction=np.full(len(ways), Direction.BOTH, dtype=np.uint8),
        oneway_corroborated=np.zeros(len(ways), dtype=np.bool_),
    )
    return way_table, nodes


_TRACK = WayTags(highway="track")
_FERRY = WayTags(route="ferry")


def test_a_ferry_does_not_drag_the_footprint_out_to_sea():
    # A crossing with one quay inside the clip box survives it, so filtering
    # by the box alone would still ask for tiles over the North Sea.
    ways, nodes = _tables(
        [(50.5, 5.5), (49.7, 5.9), (51.3, 3.2), (53.7, -0.3)],
        [([0, 1], _TRACK), ([2, 3], _FERRY)],
    )

    area = elevation_footprint(ways, nodes)

    assert (area.min_lat, area.max_lat) == (49.7, 50.5)
    assert (area.min_lon, area.max_lon) == (5.5, 5.9)


def test_coordinates_no_surviving_way_touches_are_left_out():
    ways, nodes = _tables(
        [(50.5, 5.5), (49.7, 5.9), (54.2, 3.1)],
        [([0, 1], _TRACK)],
    )

    area = elevation_footprint(ways, nodes)

    assert area.max_lat == 50.5


def test_the_footprint_stays_within_the_tiles_the_network_needs():
    ways, nodes = _tables(
        [(50.5, 5.5), (49.7, 5.9), (53.7, -0.3)],
        [([0, 1], _TRACK), ([1, 2], _FERRY)],
    )

    tiles = tiles_for_bbox(elevation_footprint(ways, nodes))

    assert {tile.name for tile in tiles} == {
        "Copernicus_DSM_COG_10_N49_00_E005_00_DEM",
        "Copernicus_DSM_COG_10_N50_00_E005_00_DEM",
    }


def test_a_quay_a_road_also_touches_still_counts():
    # The ferry is excluded, not the ground it leaves from.
    ways, nodes = _tables(
        [(50.5, 5.5), (51.3, 3.2), (53.7, -0.3)],
        [([0, 1], _TRACK), ([1, 2], _FERRY)],
    )

    assert elevation_footprint(ways, nodes).min_lon == 3.2


def test_a_network_of_nothing_but_ferries_is_an_error():
    ways, nodes = _tables([(51.3, 3.2), (53.7, -0.3)], [([0, 1], _FERRY)])

    with pytest.raises(BuildError, match="ferry"):
        elevation_footprint(ways, nodes)
