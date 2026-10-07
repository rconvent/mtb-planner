"""GPX export.

Every exported file carries the access disclaimer. OpenStreetMap does not
model Belgian access law — Flanders restricts mountain biking to designated
routes in many nature reserves, and the Wallonian forest code adds seasonal
closures OSM does not represent at all. A GPX file outlives the session that
produced it, so the caveat has to travel with it.
"""

from __future__ import annotations

from pathlib import Path

import gpxpy.gpx

from planner_core import __version__
from planner_core.routing.result import Route
from planner_core.sport import active_sport

#: Attached to every export. Deliberately blunt.
ACCESS_DISCLAIMER = (
    "Generated from OpenStreetMap data, which does not fully model access law. "
    "This is not confirmation that the route is legal to ride. "
    "Check local signage and regulations."
)

#: 1e-7 degrees is roughly a centimetre; more digits are noise in the file.
_COORD_DIGITS = 7


def _summarise(route: Route) -> str:
    """One-line human summary of a route's measurements."""
    stats = route.stats
    parts = [f"{stats.distance_km:.2f} km", f"{stats.ascent_m:.0f} m climb"]
    if stats.walk_m > 0:
        parts.append(f"{stats.walk_m:.0f} m on foot")
    if route.is_loop:
        parts.append("loop")
    return " · ".join(parts)


def route_to_gpx(route: Route, name: str, description: str | None = None) -> str:
    """Render a route as a GPX 1.1 document."""
    gpx = gpxpy.gpx.GPX()
    gpx.creator = f"{active_sport().cli}-planner {__version__}"

    summary = _summarise(route)
    full_description = f"{description}\n\n{summary}" if description else summary
    gpx.description = f"{full_description}\n\n{ACCESS_DISCLAIMER}"

    track = gpxpy.gpx.GPXTrack(name=name, description=summary)
    # gpxpy declares this attribute as a bare `None` class default.
    track.type = "cycling"  # type: ignore[assignment]
    gpx.tracks.append(track)

    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    segment.points.extend(
        gpxpy.gpx.GPXTrackPoint(
            latitude=round(float(lat), _COORD_DIGITS),
            longitude=round(float(lon), _COORD_DIGITS),
        )
        for lat, lon in route.geometry
    )

    return gpx.to_xml()


def write_gpx(route: Route, path: Path, name: str, description: str | None = None) -> Path:
    """Write a route to a `.gpx` file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(route_to_gpx(route, name=name, description=description), encoding="utf-8")
    return path
