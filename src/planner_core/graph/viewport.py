"""The trail network, as a map layer.

The planner already holds every path, track and lane in the area of interest, in
memory, with its surface and difficulty. Until now none of it reached the screen:
the map drew the planned route on blank paper, so a rider could see the shape of
a ride but not where it was, what it connected to, or what else was nearby.

This turns the graph into something a map can draw. It is not a basemap — there
are no towns, no labels, no land cover — it is a *trail* map, which is what the
plan asks for: paths and tracks first class, roads down to hairlines. The data is
the same data the router uses, so what a rider sees is exactly what can be routed
over, and there is no second extraction path to drift out of step.

Three things keep it affordable at country scale:

**Selection is a bounding-box test on endpoints**, vectorised over all nodes, the
same trick the search ball uses. An edge whose two ends both lie outside the view
but whose middle crosses it is missed; at any zoom where that matters the edge is
longer than the screen, and fixing it costs a per-edge bounding box in the
artifact, which is not worth a schema bump.

**Minor classes appear only when they are legible.** Showing every urban service
road at zoom 9 is both unreadable and a megabyte of payload.

**Geometry is decimated by zoom, and sub-pixel edges are dropped.** A vertex
every few metres is invisible above zoom 14 and most of the payload below it; an
edge shorter than a few pixels of ground is a dot whether it is sent or not, and
in a town there are thousands of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from planner_core.core.geo import BBox
from planner_core.features.enums import WayClass
from planner_core.features.model import ROUTE_SIGNED
from planner_core.graph.store import GraphStore

#: The zoom at which each way class starts being drawn. Trails are always shown
#: because they are the subject; roads arrive as they become useful for
#: orientation; urban clutter arrives last.
MIN_ZOOM: dict[int, int] = {
    int(WayClass.SINGLETRACK): 0,
    int(WayClass.DOUBLETRACK): 0,
    int(WayClass.FIREROAD): 0,
    int(WayClass.GRAVEL_ROAD): 0,
    int(WayClass.STEPS): 13,
    int(WayClass.FERRY): 0,
    int(WayClass.PAVED_MAJOR): 10,
    int(WayClass.PAVED_MINOR): 12,
    int(WayClass.URBAN): 14,
    int(WayClass.UNKNOWN): 13,
}

#: Vertex stride by zoom. Below zoom 11 a trail is a few pixels wide and its
#: wiggle cannot be seen; above 14 every vertex is worth having.
_STRIDE = ((11, 6), (13, 3), (15, 2))

#: Most features to return, whatever the view. A viewport can legitimately hold
#: tens of thousands of edges; a browser cannot usefully draw them, and the
#: payload would dwarf every other request the app makes.
DEFAULT_LIMIT = 6000

#: An edge shorter than this many pixels on screen cannot be told from a dot, so
#: it is not worth its share of the payload. Four is conservative: it keeps
#: anything that reads as a line.
MIN_PIXELS = 4.0

#: Ground resolution of one pixel at zoom 0 on the equator, in metres.
_METRES_PER_PIXEL_Z0 = 156543.03

#: Five decimals is about a metre, which is finer than a hairline can express and
#: noticeably cheaper than the six the route payload uses: this is the largest
#: response the app makes, and it is refetched on every pan.
_DIGITS = 5


@dataclass(frozen=True)
class NetworkView:
    """What was drawn, and what was left out."""

    geojson: dict[str, Any]

    total: int
    """Edges inside the view and above their zoom threshold."""

    drawn: int
    """Edges actually returned."""

    @property
    def truncated(self) -> bool:
        """Whether the limit bit.

        Reported rather than hidden: a silently thinned map looks like a gap in
        the trail network, which is exactly the wrong thing to make a rider
        believe.
        """
        return self.drawn < self.total


def stride_for(zoom: int) -> int:
    """How many vertices to skip at this zoom."""
    for threshold, stride in _STRIDE:
        if zoom < threshold:
            return stride
    return 1


def visible_classes(zoom: int) -> tuple[int, ...]:
    """Which way classes are drawn at this zoom."""
    return tuple(sorted(cls for cls, lowest in MIN_ZOOM.items() if zoom >= lowest))


def min_length_m(zoom: int, latitude: float) -> float:
    """Shortest edge worth sending at this zoom.

    Web Mercator ground resolution, so the threshold follows what a pixel
    actually covers rather than a guess per zoom level.
    """
    metres_per_pixel = _METRES_PER_PIXEL_Z0 * math.cos(math.radians(latitude)) / (2.0**zoom)
    return MIN_PIXELS * metres_per_pixel


def select_edges(store: GraphStore, bbox: BBox, zoom: int) -> npt.NDArray[np.intp]:
    """Edges worth drawing in this view, longest first.

    Ordered by length so that truncating the result drops the least visible
    edges rather than an arbitrary corner of the map.
    """
    lat = store.node_lat
    lon = store.node_lon

    inside = (
        (lat >= bbox.min_lat)
        & (lat <= bbox.max_lat)
        & (lon >= bbox.min_lon)
        & (lon <= bbox.max_lon)
    )
    touching = inside[store.edge_u] | inside[store.edge_v]

    allowed = np.isin(store.features.way_class, visible_classes(zoom))

    middle = (bbox.min_lat + bbox.max_lat) / 2.0
    long_enough = store.edge_length_m >= min_length_m(zoom, middle)

    candidates = np.flatnonzero(touching & allowed & long_enough)

    order = np.argsort(-store.edge_length_m[candidates], kind="stable")
    return candidates[order].astype(np.intp)


def as_geojson(
    store: GraphStore,
    edges: npt.NDArray[np.intp],
    zoom: int,
    limit: int = DEFAULT_LIMIT,
) -> NetworkView:
    """Render selected edges as GeoJSON, longitude first."""
    stride = stride_for(zoom)
    features: list[dict[str, Any]] = []
    way_class = store.features.way_class
    route_flags = store.features.route_flags

    for edge in edges[:limit]:
        points = store.edge_geometry(int(edge))
        if points.shape[0] < 2:
            continue

        if stride > 1 and points.shape[0] > 2:
            kept = points[::stride]
            # Never drop an endpoint: a decimated trail that stops short of its
            # junction looks like a dead end.
            if not np.array_equal(kept[-1], points[-1]):
                kept = np.vstack((kept, points[-1]))
        else:
            kept = points

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "class": int(way_class[edge]),
                    "signed": bool(int(route_flags[edge]) & ROUTE_SIGNED),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [round(float(lon), _DIGITS), round(float(lat), _DIGITS)]
                        for lat, lon in kept
                    ],
                },
            }
        )

    return NetworkView(
        geojson={"type": "FeatureCollection", "features": features},
        total=int(edges.shape[0]),
        drawn=len(features),
    )


def network_view(
    store: GraphStore,
    bbox: BBox,
    zoom: int,
    limit: int = DEFAULT_LIMIT,
) -> NetworkView:
    """The trail network inside a bounding box, ready to draw."""
    return as_geojson(store, select_edges(store, bbox, zoom), zoom, limit)


def tile_zoom(bbox: BBox, width_px: int = 1200) -> int:
    """Approximate the zoom a viewport of this span is being shown at.

    Used when a client reports its bounds but not its zoom, so the decimation
    and class filtering still match what can actually be seen.
    """
    span = max(1e-6, bbox.max_lon - bbox.min_lon)
    return int(max(0, min(20, round(math.log2(360.0 * width_px / (256.0 * span))))))
