"""The trail network as map data.

This is the one endpoint that does **not** answer in the standard envelope. It
returns bare GeoJSON, because MapLibre consumes a GeoJSON source by URL and
wrapping it would mean the front end fetching, unwrapping and re-handing the
payload over for no gain. What was drawn and what was left out travel in headers
instead, so a client can still tell when a view was thinned.

Nothing here is a second source of truth: the geometry, the way classes and the
route flags are the ones the router uses, read out of the same loaded graph.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import JSONResponse

from mtb_planner.api.deps import require_graph
from mtb_planner.core.errors import PointOutsideCoverageError
from mtb_planner.core.geo import BBox
from mtb_planner.graph.viewport import DEFAULT_LIMIT, network_view
from mtb_planner.services.graph_registry import LoadedGraph

router = APIRouter(tags=["map"])

#: Widest span served in one request. A whole-country view is neither drawable
#: nor useful, and asking for one is a sign a client has lost its bounds.
MAX_SPAN_DEGREES = 6.0

#: Browsers refetch this on every pan. It is derived from an artifact that only
#: changes when the graph is rebuilt, so it is safe to keep for a while.
CACHE_SECONDS = 3600


@router.get("/api/network")
def trail_network(
    west: float = Query(ge=-180.0, le=180.0),
    south: float = Query(ge=-90.0, le=90.0),
    east: float = Query(ge=-180.0, le=180.0),
    north: float = Query(ge=-90.0, le=90.0),
    zoom: int = Query(default=12, ge=0, le=22),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=20_000),
    graph: LoadedGraph = Depends(require_graph),
) -> Response:
    """The trails inside a bounding box, as GeoJSON.

    Bare GeoJSON rather than the usual envelope, so MapLibre can point a source
    straight at it.
    """
    if east <= west or north <= south:
        raise PointOutsideCoverageError(lat=north, lon=east)

    if (east - west) > MAX_SPAN_DEGREES or (north - south) > MAX_SPAN_DEGREES:
        # Refused rather than clipped: a client that asks for a continent has a
        # bug, and quietly serving it a corner of one hides that.
        raise PointOutsideCoverageError(lat=north, lon=east)

    view = network_view(
        graph.store,
        BBox(min_lat=south, min_lon=west, max_lat=north, max_lon=east),
        zoom=zoom,
        limit=limit,
    )

    return JSONResponse(
        content=view.geojson,
        headers={
            "Cache-Control": f"public, max-age={CACHE_SECONDS}",
            "X-Network-Total": str(view.total),
            "X-Network-Drawn": str(view.drawn),
            "X-Network-Truncated": "1" if view.truncated else "0",
        },
    )
