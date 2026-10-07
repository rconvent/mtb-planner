"""The map style, assembled per request.

`paper.json` holds the design: the base document a designer edits. This endpoint
adds whatever the deployment is configured to put under it, which is currently a
raster basemap for orientation.

Built here rather than baked into the file because the basemap is a *choice*.
Everything else in this project runs from a local artifact with nothing in the
request path, and drawing tiles from a third party gives that up. Keeping it in
settings means the trade is made once, visibly, and can be undone by setting
`MTB_BASEMAP_URL` to an empty string — which leaves the paper map the rest of the
design assumes.

The basemap is deliberately washed out. Its job is to say which valley you are
looking at, not to compete with the route: the difficulty ramp is the only
saturated thing on screen, and a full-strength OSM raster underneath it would
turn the whole plate into noise.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from planner_core.api.routes.pages import STATIC_ROOT
from planner_core.config.settings import Settings, get_settings

router = APIRouter(tags=["map"])

#: Identifier for the injected source and layer. Named so it reads clearly in
#: the MapLibre inspector next to `trail-network` and `route-segments`.
BASEMAP = "basemap"

#: How far the basemap is pushed back. Saturation does most of the work: OSM
#: renders greens and reds at full strength, and the difficulty ramp needs to be
#: the only saturated thing on the map.
BASEMAP_PAINT: dict[str, Any] = {
    "raster-opacity": 0.55,
    "raster-saturation": -0.6,
    "raster-contrast": -0.1,
}

#: Tiles change rarely and the style is tiny, but it is fetched on every load.
CACHE_SECONDS = 300


@lru_cache(maxsize=1)
def _base_style() -> dict[str, Any]:
    """The design document, read once."""
    path = STATIC_ROOT / "style" / "paper.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def build_style(settings: Settings) -> dict[str, Any]:
    """The base style with the configured basemap underneath it.

    The basemap layer goes first, so everything added at run time — the trail
    network, then the route — draws over it without having to name a `beforeId`.
    """
    style: dict[str, Any] = json.loads(json.dumps(_base_style()))

    if not settings.basemap_enabled:
        return style

    style["sources"][BASEMAP] = {
        "type": "raster",
        "tiles": [settings.basemap_url],
        "tileSize": 256,
        "maxzoom": settings.basemap_max_zoom,
        # Most providers require this, and MapLibre surfaces it in the
        # attribution control automatically.
        "attribution": settings.basemap_attribution,
    }
    style["layers"] = [
        {"id": BASEMAP, "type": "raster", "source": BASEMAP, "paint": BASEMAP_PAINT},
        *style["layers"],
    ]
    return style


@router.get("/api/style")
def map_style() -> Response:
    """The MapLibre style for the interface.

    Bare style JSON rather than the usual envelope, for the same reason
    `/api/network` is: MapLibre consumes it by URL.
    """
    return JSONResponse(
        content=build_style(get_settings()),
        headers={"Cache-Control": f"public, max-age={CACHE_SECONDS}"},
    )
