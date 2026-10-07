"""Named synthetic networks.

`ardennes_like` is the workhorse. It is small enough to reason about by hand
and deliberately contains every case the ingest and graph layers must get
right: a legally closed way, a way bicycles are banned from, a corroborated
one-way descent, a pushing section, an isolated fragment, and a signed route
relation.

Layout of the rideable grid, 0.001° apart (about 111 m north-south, 71 m
east-west at this latitude):

    1 --- 2 --- 3        lat 50.102
    |     |     |
    4 --- 5 --- 6        lat 50.101
    |     |     |
    7 --- 8 --- 9        lat 50.100

  lon  5.100 5.101 5.102

Plus, hanging off it:
  * way 202: a one-way descent 9 -> 13 -> 3, tagged `mtb:scale=3`, so the
    one-way is corroborated and genuinely directional.
  * way 200: a private driveway from node 3. Must never appear in a route.
  * way 203: a `bicycle=no` path from node 5. Must never appear in a route.
  * way 204: steps from node 1. Rideable only on foot.
  * way 201: an unconnected fragment 11 km north. Its own component.
"""

from __future__ import annotations

from pathlib import Path

from tests.fixtures.osm_builder import (
    FixtureNode,
    FixtureRelation,
    FixtureWay,
    write_fixture_pbf,
)

#: The grid corner used as a routing start in tests.
GRID_NW = (50.102, 5.100)
#: The diagonally opposite corner.
GRID_SE = (50.100, 5.102)
#: Somewhere on the isolated fragment.
ISLAND = (50.200, 5.200)
#: Far outside any fixture network.
FAR_AWAY = (48.000, 2.000)

_TRACK = {"highway": "track", "surface": "gravel", "tracktype": "grade3"}

_GRID_NODES = [
    FixtureNode(1, 50.102, 5.100),
    FixtureNode(2, 50.102, 5.101),
    FixtureNode(3, 50.102, 5.102),
    FixtureNode(4, 50.101, 5.100),
    FixtureNode(5, 50.101, 5.101),
    FixtureNode(6, 50.101, 5.102),
    FixtureNode(7, 50.100, 5.100),
    FixtureNode(8, 50.100, 5.101),
    FixtureNode(9, 50.100, 5.102),
]

_EXTRA_NODES = [
    FixtureNode(10, 50.1025, 5.1030),  # end of the private driveway
    FixtureNode(11, 50.200, 5.200),  # isolated fragment
    FixtureNode(12, 50.201, 5.201),
    FixtureNode(13, 50.101, 5.1025),  # interior point of the descent
    FixtureNode(14, 50.1015, 5.1015),  # end of the bicycle=no path
    FixtureNode(15, 50.1025, 5.0995),  # top of the steps
]

_WAYS = [
    # Grid: three rows, then three columns.
    FixtureWay(101, [1, 2, 3], _TRACK),
    FixtureWay(102, [4, 5, 6], _TRACK),
    FixtureWay(103, [7, 8, 9], _TRACK),
    FixtureWay(104, [1, 4, 7], _TRACK),
    FixtureWay(105, [2, 5, 8], _TRACK),
    FixtureWay(106, [3, 6, 9], _TRACK),
    # A private driveway. Legally closed, so it must never be routed.
    FixtureWay(
        200,
        [3, 10],
        {"highway": "service", "service": "driveway", "access": "private"},
    ),
    # An unconnected fragment, to exercise the connectivity filter.
    FixtureWay(201, [11, 12], {"highway": "path", "surface": "ground"}),
    # A descent that is genuinely one-way: the mtb:scale corroborates it.
    FixtureWay(
        202,
        [9, 13, 3],
        {"highway": "path", "oneway": "yes", "mtb:scale": "3", "surface": "dirt"},
    ),
    # Bicycles banned outright.
    FixtureWay(203, [5, 14], {"highway": "path", "bicycle": "no"}),
    # Rideable only by pushing.
    FixtureWay(204, [1, 15], {"highway": "steps", "step_count": "18"}),
]

_RELATIONS = [
    FixtureRelation(
        900,
        [("way", 101, ""), ("way", 106, "")],
        {"type": "route", "route": "mtb", "name": "Rode lus", "network": "lcn"},
    ),
    FixtureRelation(
        901,
        [("way", 102, "")],
        {"type": "route", "route": "bicycle", "name": "Knooppunt 42"},
    ),
]


def ardennes_like(directory: Path, stem: str = "ardennes_like") -> Path:
    """Write the full fixture network and return the `.osm.pbf` path."""
    return write_fixture_pbf(
        directory,
        stem,
        nodes=[*_GRID_NODES, *_EXTRA_NODES],
        ways=_WAYS,
        relations=_RELATIONS,
    )


#: Side length of the lattice network, in nodes.
LATTICE_SIZE = 9

#: Lattice spacing. About 444 m north-south and 420 m east-west at 51°N, so the
#: cells are roughly square and a single cell is a ~1.7 km loop.
_LATTICE_DLAT = 0.004
_LATTICE_DLON = 0.006

#: South-west corner of the lattice, well away from every other fixture.
_LATTICE_LAT0 = 51.000
_LATTICE_LON0 = 4.000

#: The middle of the lattice, used as a loop start.
LATTICE_CENTRE = (
    _LATTICE_LAT0 + _LATTICE_DLAT * (LATTICE_SIZE // 2),
    _LATTICE_LON0 + _LATTICE_DLON * (LATTICE_SIZE // 2),
)

#: Opposite corners of the lattice, used as point-to-point endpoints. Far
#: enough apart that many routes connect them, which is what a test about
#: choosing between routes needs.
LATTICE_SW = (_LATTICE_LAT0, _LATTICE_LON0)
LATTICE_NE = (
    _LATTICE_LAT0 + _LATTICE_DLAT * (LATTICE_SIZE - 1),
    _LATTICE_LON0 + _LATTICE_DLON * (LATTICE_SIZE - 1),
)

#: Way tags cycled through the lattice, so preference, technicality and surface
#: all vary across it. A uniform lattice would make every loop equally good and
#: leave scoring and diversification with nothing to distinguish.
_LATTICE_TAGS = (
    {"highway": "path", "surface": "dirt", "mtb:scale": "2", "width": "0.8"},
    {"highway": "track", "surface": "gravel", "tracktype": "grade3"},
    {"highway": "path", "surface": "ground", "mtb:scale": "1"},
    {"highway": "unclassified", "surface": "asphalt"},
)


def lattice(directory: Path, stem: str = "lattice", size: int = LATTICE_SIZE) -> Path:
    """A square lattice big enough to hold real loops.

    `ardennes_like` is 1.5 km of trail in total, which is too little to satisfy
    the smallest time budget the API accepts — so it cannot exercise the loop
    generator's success path at all. This network is about 60 km of trail in a
    3.5 km square: enough for a fifteen-minute loop many times over, and enough
    distinct cycles that sector spreading and diversification have something to
    choose between.

    Each row and each column is one way spanning the whole lattice, so the graph
    builder still has to split them at every junction.
    """
    nodes = [
        FixtureNode(
            1000 + row * size + column,
            _LATTICE_LAT0 + _LATTICE_DLAT * row,
            _LATTICE_LON0 + _LATTICE_DLON * column,
        )
        for row in range(size)
        for column in range(size)
    ]

    ways: list[FixtureWay] = [
        FixtureWay(
            300 + row,
            [1000 + row * size + column for column in range(size)],
            _LATTICE_TAGS[row % len(_LATTICE_TAGS)],
        )
        for row in range(size)
    ]
    ways.extend(
        FixtureWay(
            400 + column,
            [1000 + row * size + column for row in range(size)],
            _LATTICE_TAGS[(column + 2) % len(_LATTICE_TAGS)],
        )
        for column in range(size)
    )

    return write_fixture_pbf(directory, stem, nodes=nodes, ways=ways)


def single_track(directory: Path, stem: str = "single_track") -> Path:
    """A three-node path with no junctions — the simplest routable network."""
    return write_fixture_pbf(
        directory,
        stem,
        nodes=[
            FixtureNode(1, 50.100, 5.100),
            FixtureNode(2, 50.101, 5.101),
            FixtureNode(3, 50.102, 5.102),
        ],
        ways=[FixtureWay(1, [1, 2, 3], {"highway": "path", "surface": "ground"})],
    )


def empty_of_ways(directory: Path, stem: str = "no_ways") -> Path:
    """Nodes but nothing rideable, to exercise the empty-extract failure."""
    return write_fixture_pbf(
        directory,
        stem,
        nodes=[FixtureNode(1, 50.100, 5.100), FixtureNode(2, 50.101, 5.101)],
        ways=[FixtureWay(1, [1, 2], {"highway": "motorway"})],
    )


def border_pair(directory: Path) -> tuple[Path, Path]:
    """Two overlapping extracts sharing a way, as adjacent regions would.

    The western file holds the way truncated at the border; the eastern file
    holds it whole. Merging must keep the longer copy, or the network acquires
    a wall exactly along the frontier.
    """
    west_nodes = [
        FixtureNode(1, 50.100, 5.100),
        FixtureNode(2, 50.100, 5.101),
    ]
    east_nodes = [
        FixtureNode(1, 50.100, 5.100),
        FixtureNode(2, 50.100, 5.101),
        FixtureNode(3, 50.100, 5.102),
        FixtureNode(4, 50.100, 5.103),
    ]
    tags = {"highway": "track", "tracktype": "grade2"}

    west = write_fixture_pbf(
        directory,
        "west",
        nodes=west_nodes,
        ways=[FixtureWay(500, [1, 2], tags)],
        relations=[
            FixtureRelation(
                950,
                [("way", 500, "")],
                {"type": "route", "route": "mtb", "name": "Cross-border"},
            )
        ],
    )
    east = write_fixture_pbf(
        directory,
        "east",
        nodes=east_nodes,
        ways=[FixtureWay(500, [1, 2, 3, 4], tags)],
        relations=[
            FixtureRelation(
                950,
                [("way", 500, ""), ("way", 501, "")],
                {"type": "route", "route": "mtb", "name": "Cross-border"},
            )
        ],
    )
    return west, east


#: Corners of the footpath grid, for routing in the trail tests.
FOOT_SW = (50.300, 5.300)
FOOT_NE = (50.302, 5.302)


def footpaths(directory: Path, stem: str = "footpaths") -> Path:
    """A grid where a runner and a rider must disagree.

    Laid out like `ardennes_like` — 3x3, 0.001° apart — but at lat 50.30 so it
    can never be confused with it, and with the cases that split the sports:

      * way 510: a footway with `bicycle=no`. A runner's; never a rider's.
      * way 511: a cycleway with `foot=no`. A rider's; never a runner's.
      * way 512: a one-way mountain-bike descent. One-way on a bike, both ways
        on foot.
      * way 513: steps. Pushed on a bike, run on foot.
      * way 514: a `demanding_alpine_hiking` scramble. Walked on foot.
      * way 515: a `sac_scale=mountain_hiking` path. Surveyed for a runner.
      * relation 950: a waymarked hiking route over 500 and 515.
    """
    lat0, lon0, step = 50.300, 5.300, 0.001
    grid = [
        FixtureNode(2000 + row * 3 + column, lat0 + step * row, lon0 + step * column)
        for row in range(3)
        for column in range(3)
    ]
    extra = [
        FixtureNode(2010, lat0 + 0.0005, lon0 - 0.0008),
        FixtureNode(2011, lat0 + 0.0025, lon0 + 0.0010),
        FixtureNode(2012, lat0 + 0.0015, lon0 + 0.0028),
    ]

    track = {"highway": "track", "surface": "gravel", "tracktype": "grade3"}
    ways = [
        FixtureWay(500, [2000, 2001, 2002], track),
        FixtureWay(501, [2003, 2004, 2005], track),
        FixtureWay(502, [2006, 2007, 2008], track),
        FixtureWay(503, [2000, 2003, 2006], track),
        FixtureWay(504, [2002, 2005, 2008], track),
        FixtureWay(510, [2001, 2004], {"highway": "footway", "bicycle": "no", "surface": "ground"}),
        FixtureWay(511, [2004, 2007], {"highway": "cycleway", "foot": "no", "surface": "asphalt"}),
        FixtureWay(
            512,
            [2008, 2012, 2005],
            {"highway": "path", "oneway": "yes", "mtb:scale": "2", "surface": "dirt"},
        ),
        FixtureWay(513, [2000, 2010], {"highway": "steps", "step_count": "40"}),
        FixtureWay(
            514,
            [2006, 2011],
            {"highway": "path", "sac_scale": "demanding_alpine_hiking", "surface": "rock"},
        ),
        FixtureWay(
            515,
            [2003, 2010],
            {"highway": "path", "sac_scale": "mountain_hiking", "surface": "ground"},
        ),
    ]
    relations = [
        FixtureRelation(
            950,
            [("way", 500, ""), ("way", 515, "")],
            {"type": "route", "route": "hiking", "name": "GR 5", "network": "nwn"},
        ),
    ]
    return write_fixture_pbf(
        directory, stem, nodes=[*grid, *extra], ways=ways, relations=relations
    )
