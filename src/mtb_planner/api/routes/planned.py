"""Reading back a route that has already been planned.

A browser cannot trigger a file download from a POST body without blob
gymnastics, so a planned route gets an id and these endpoints serve it from the
cache behind a plain link.

Ids are validated against `ROUTE_ID_PATTERN`. Without that, `{route_id}`
matches anything and `/api/routes/loop` would be answered as an expired route —
telling a rider to re-plan something that never existed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Response

from mtb_planner.api.deps import require_graph, route_cache
from mtb_planner.api.schemas.common import ApiResponse, LatLonModel
from mtb_planner.api.schemas.planning import (
    RouteModel,
    SnapModel,
    character_of,
    elevation_of,
    geometry_of,
    segments_of,
    stats_of,
)
from mtb_planner.export.gpx import route_to_gpx
from mtb_planner.services.graph_registry import LoadedGraph
from mtb_planner.services.route_cache import ROUTE_ID_PATTERN, RouteCache

router = APIRouter(tags=["planning"])

#: Path parameter spec shared by both endpoints.
RouteId = Path(pattern=ROUTE_ID_PATTERN, description="Id of a previously planned route.")


@router.get("/api/routes/{route_id}/gpx")
def download_gpx(
    route_id: str = RouteId,
    cache: RouteCache = Depends(route_cache),
) -> Response:
    """Download a previously planned route.

    A plain link rather than a POST body, so a browser can simply follow it.
    """
    entry = cache.get(route_id)
    body = route_to_gpx(
        entry.route,
        name=f"{entry.profile_name} route",
        description=f"Planned with the {entry.profile_name} profile",
    )
    return Response(
        content=body,
        media_type="application/gpx+xml",
        headers={"Content-Disposition": f'attachment; filename="{entry.filename}"'},
    )


@router.get("/api/routes/{route_id}", response_model=ApiResponse[RouteModel])
def get_route(
    route_id: str = RouteId,
    cache: RouteCache = Depends(route_cache),
    graph: LoadedGraph = Depends(require_graph),
) -> ApiResponse[RouteModel]:
    """Re-read a planned route without re-planning it."""
    entry = cache.get(route_id)
    return ApiResponse.ok(
        RouteModel(
            route_id=entry.route_id,
            geometry=geometry_of(entry.route),
            stats=stats_of(entry.route),
            segments=segments_of(graph.store, entry.route),
            elevation=elevation_of(graph.store, entry.route),
            start_snap=SnapModel(point=LatLonModel.from_domain(entry.route.start), distance_m=0.0),
            end_snap=SnapModel(point=LatLonModel.from_domain(entry.route.end), distance_m=0.0),
            gpx_url=f"/api/routes/{entry.route_id}/gpx",
            character=None if entry.metrics is None else character_of(entry.metrics),
            overlap_ratio=(
                None if entry.metrics is None else round(entry.metrics.overlap_ratio, 3)
            ),
            score=entry.score,
            within_budget=entry.within_budget,
        ),
        profile=entry.profile_id,
    )
