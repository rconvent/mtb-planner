"""Which ground a build has to have elevation for.

Deriving this from every ingested coordinate asks for far more of the world
than the network occupies. Two things widen the node table beyond the riding:

* Ways the clip box discarded keep their coordinates. The table is not
  rebuilt when ways are dropped, only re-counted.
* Ferries. They carry bikes, so the way filter admits them, and Geofabrik
  ships them whole — the Belgian extract alone reaches the Channel Islands
  and the middle of the North Sea. A ferry with one quay inside the clip box
  survives it, dragging the box out to sea with it.

GLO-30 publishes land, so measuring open water in demands tiles that do not
exist and stops the build over ground nobody rides. A ferry crossing needs no
elevation of its own: it is flat by nature, and where no tile covers it the
sampler already reports the edge flat and says how many it did that to.
"""

from __future__ import annotations

import numpy as np

from mtb_planner.core.errors import BuildError
from mtb_planner.core.geo import BBox
from mtb_planner.ingest.tables import NodeTable, WayTable
from mtb_planner.ingest.tags import first_semicolon_value

#: `route=ferry` is how a crossing is tagged; the way filter admits those on
#: the strength of it, so the same test identifies them again here.
_FERRY = "ferry"


def _ferry_mask(ways: WayTable) -> np.ndarray:
    """Which ways are ferry crossings rather than ground."""
    return np.fromiter(
        (first_semicolon_value(tags.route) == _FERRY for tags in ways.tags),
        dtype=np.bool_,
        count=len(ways),
    )


def elevation_footprint(ways: WayTable, nodes: NodeTable) -> BBox:
    """The area whose DEM tiles a build actually needs.

    Spans the nodes of every surviving way except the ferries. Raises rather
    than returning an empty box: nothing downstream can do anything useful
    with a network of no ground at all.
    """
    if len(ways) == 0:
        msg = "No way survived ingest, so there is no ground to cover"
        raise BuildError(msg)

    on_land = ~_ferry_mask(ways)
    refs = ways.node_refs[np.repeat(on_land, np.diff(ways.node_offset))]
    if refs.shape[0] == 0:
        msg = "Every surviving way is a ferry, so there is no ground to cover"
        raise BuildError(msg)

    lat, lon = nodes.degrees()
    positions = np.unique(nodes.index_of(refs))
    lat, lon = lat[positions], lon[positions]

    return BBox(
        min_lon=float(lon.min()),
        min_lat=float(lat.min()),
        max_lon=float(lon.max()),
        max_lat=float(lat.max()),
    )
