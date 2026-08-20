"""Parsing of command line geographic arguments."""

from __future__ import annotations

import typer

from mtb_planner.core.geo import BBox, LatLon


def parse_latlon(value: str, label: str = "coordinate") -> LatLon:
    """Parse a `lat,lon` pair in decimal degrees.

    Raises `typer.BadParameter` rather than a domain error: this is a CLI
    input problem, and the user needs the usage hint, not a stack trace.
    """
    parts = value.replace(" ", "").split(",")
    if len(parts) != 2:
        msg = f"{label} must be 'lat,lon' in decimal degrees, got {value!r}"
        raise typer.BadParameter(msg)
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError as exc:
        msg = f"{label} must be two numbers, got {value!r}"
        raise typer.BadParameter(msg) from exc

    if not -90.0 <= lat <= 90.0:
        msg = f"{label} latitude {lat} is outside -90..90"
        raise typer.BadParameter(msg)
    if not -180.0 <= lon <= 180.0:
        msg = f"{label} longitude {lon} is outside -180..180"
        raise typer.BadParameter(msg)
    return LatLon(lat=lat, lon=lon)


def parse_bbox(value: str | None) -> BBox | None:
    """Parse a `min_lon,min_lat,max_lon,max_lat` bounding box.

    Argument order matches GeoJSON and the Geofabrik site, so a box can be
    copied from either without rearrangement.
    """
    if value is None:
        return None
    parts = value.replace(" ", "").split(",")
    if len(parts) != 4:
        msg = f"bbox must be 'min_lon,min_lat,max_lon,max_lat', got {value!r}"
        raise typer.BadParameter(msg)
    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError as exc:
        msg = f"bbox must be four numbers, got {value!r}"
        raise typer.BadParameter(msg) from exc

    if min_lon >= max_lon or min_lat >= max_lat:
        msg = "bbox minimums must be smaller than its maximums"
        raise typer.BadParameter(msg)
    return BBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)
