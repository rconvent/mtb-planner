"""The app shell, and the static assets it loads.

One HTML page. It carries no route data: the shell renders, then asks the API
for profiles, health and a plan like any other client would. That keeps exactly
one path into the planner instead of a server-rendered shortcut that drifts from
the JSON everyone else sees.

A content security policy is set here rather than left to a proxy, because the
one thing this page must not do is fetch code from somewhere else. Vendored
MapLibre carries a subresource integrity hash, so a tampered file fails to
execute rather than running quietly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from planner_core.config.settings import get_settings
from planner_core.sport import active_sport

router = APIRouter(tags=["web"])

WEB_ROOT = Path(__file__).resolve().parent.parent.parent / "web"
STATIC_ROOT = WEB_ROOT / "static"
TEMPLATE_ROOT = WEB_ROOT / "templates"


def content_security_policy(basemap_host: str = "") -> str:
    """The policy for the app shell.

    `script-src` stays `self` unconditionally: code comes from this server or it
    does not run, and that is the half of the policy that matters. A configured
    basemap adds exactly one origin, derived from the setting rather than written
    out: a policy naming a different host from the one actually fetched blocks
    every tile and leaves a blank map with nothing but a console error.

    That origin has to appear in **both** `img-src` and `connect-src`. MapLibre
    pulls raster tiles through the Fetch API rather than as `<img>` elements, so
    `img-src` alone silently blocks all of them; `img-src` is kept as well
    because nothing guarantees it stays that way.

    `style-src` allows inline because MapLibre sets element styles as it renders.
    """
    images = ["'self'", "data:", "blob:"]
    connects = ["'self'"]
    if basemap_host:
        images.append(basemap_host)
        connects.append(basemap_host)

    return "; ".join(
        (
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",
            f"img-src {' '.join(images)}",
            "font-src 'self'",
            f"connect-src {' '.join(connects)}",
            "worker-src 'self' blob:",
            "frame-ancestors 'none'",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        )
    )


def security_headers(basemap_host: str = "") -> dict[str, str]:
    """Headers that cost nothing and remove whole classes of mistake."""
    return {
        "Content-Security-Policy": content_security_policy(basemap_host),
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=(self)",
    }


@lru_cache(maxsize=1)
def templates() -> Jinja2Templates:
    """The Jinja environment, built once."""
    return Jinja2Templates(directory=str(TEMPLATE_ROOT))


@lru_cache(maxsize=1)
def vendor_manifest() -> dict[str, Any]:
    """Integrity hashes for the vendored assets.

    Written by the vendoring step beside the files themselves, so the hash in
    the page always describes the bytes actually on disk. A missing manifest
    yields empty hashes rather than a failure: the page still works, it just
    loses the integrity check, and a test asserts the hash is present so the
    loss cannot pass unnoticed.
    """
    path = STATIC_ROOT / "vendor" / "manifest.json"
    if not path.exists():
        return {"maplibre_version": "unknown", "files": {}}
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def vendor_context() -> dict[str, str]:
    """What the template needs to load the vendored bundle safely."""
    manifest = vendor_manifest()
    files = manifest.get("files", {})
    return {
        "maplibre_version": str(manifest.get("maplibre_version", "unknown")),
        "maplibre_js_sri": str(files.get("maplibre-gl.js", {}).get("sri", "")),
    }


@router.get("/", response_class=HTMLResponse)
def app_shell(request: Request) -> HTMLResponse:
    """The planner interface."""
    response: HTMLResponse = templates().TemplateResponse(
        request=request,
        name="app.html",
        context={"vendor": vendor_context(), "sport": active_sport().copy},
    )
    for header, value in security_headers(get_settings().basemap_host).items():
        response.headers[header] = value
    return response
