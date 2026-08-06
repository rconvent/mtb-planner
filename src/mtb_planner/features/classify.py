"""What shape a way is, and how exposed to traffic.

Singletrack is the thing riders ask for by name, and OSM never states it
directly. It is inferred from way class, width, and whether the surface looks
engineered — a `highway=path` with no width and an unsealed surface is almost
always singletrack, while the same class with `compacted` is often a walking
path through a park.
"""

from __future__ import annotations

from mtb_planner.features.enums import SurfaceClass, WayClass
from mtb_planner.ingest.tags import WayTags, first_semicolon_value, parse_width_m

#: Above this, a track is a road wide enough for a forestry vehicle.
FIREROAD_MIN_WIDTH_M = 2.5

#: At or below this, a way admits one rider at a time.
SINGLETRACK_MAX_WIDTH_M = 1.0

#: Width beyond which singletrack likelihood has decayed to nothing.
_WIDTH_DECAY_M = 4.0

#: `tracktype` grades that still ride like a trail rather than a road.
ROUGH_TRACKTYPES: frozenset[str] = frozenset({"grade4", "grade5"})

#: Way classes that are trails first and foremost.
TRAIL_HIGHWAYS: frozenset[str] = frozenset({"path", "bridleway", "footway"})

#: Way classes carrying general motor traffic at urban speeds.
_URBAN_HIGHWAYS: frozenset[str] = frozenset(
    {"residential", "living_street", "service", "pedestrian"}
)
_MAJOR_HIGHWAYS: frozenset[str] = frozenset(
    {"secondary", "secondary_link", "primary", "primary_link"}
)
_MINOR_HIGHWAYS: frozenset[str] = frozenset({"unclassified", "tertiary", "tertiary_link", "road"})

#: How exposed to motor traffic each way class is, from 0 to 1.
TRAFFIC_EXPOSURE: dict[str, float] = {
    "path": 0.0,
    "bridleway": 0.0,
    "footway": 0.0,
    "steps": 0.0,
    "track": 0.05,
    "cycleway": 0.05,
    "pedestrian": 0.1,
    "living_street": 0.25,
    "residential": 0.35,
    "service": 0.35,
    "unclassified": 0.5,
    "road": 0.5,
    "tertiary": 0.6,
    "tertiary_link": 0.6,
    "secondary": 0.8,
    "secondary_link": 0.8,
    "primary": 1.0,
    "primary_link": 1.0,
}

#: Singletrack likelihood for each recognised situation.
_NARROW = 1.0
_BARE_TRAIL = 0.9
_TAGGED_TRAIL = 0.85
_ENGINEERED_TRAIL = 0.6
_ROUGH_TRACK = 0.3
_MIDDLING_TRACK = 0.15


def _classify_track(tags: WayTags) -> WayClass:
    """Split tracks into fire roads and the rougher doubletrack."""
    tracktype = first_semicolon_value(tags.tracktype)
    width = parse_width_m(tags.width)

    if tracktype in ROUGH_TRACKTYPES:
        return WayClass.DOUBLETRACK
    if width is not None and width < FIREROAD_MIN_WIDTH_M:
        return WayClass.DOUBLETRACK
    return WayClass.FIREROAD


def classify_way(tags: WayTags, surface_class: SurfaceClass) -> WayClass:
    """Decide what kind of way an edge is."""
    highway = first_semicolon_value(tags.highway)

    if highway is None:
        return WayClass.FERRY if first_semicolon_value(tags.route) == "ferry" else WayClass.UNKNOWN
    if highway == "steps":
        return WayClass.STEPS

    if highway in TRAIL_HIGHWAYS:
        width = parse_width_m(tags.width)
        if width is not None and width > FIREROAD_MIN_WIDTH_M:
            return WayClass.DOUBLETRACK
        return WayClass.SINGLETRACK

    if highway == "track":
        return _classify_track(tags)

    if highway == "cycleway":
        return WayClass.PAVED_MINOR if surface_class == SurfaceClass.PAVED else WayClass.GRAVEL_ROAD

    if highway in _URBAN_HIGHWAYS:
        return WayClass.URBAN
    if highway in _MAJOR_HIGHWAYS:
        return WayClass.PAVED_MAJOR
    if highway in _MINOR_HIGHWAYS:
        return WayClass.PAVED_MINOR

    return WayClass.UNKNOWN


def singletrack_score(tags: WayTags, surface_class: SurfaceClass) -> float:
    """How likely an edge is to ride as singletrack, from 0 to 1."""
    highway = first_semicolon_value(tags.highway)

    width = parse_width_m(tags.width)
    if width is not None:
        if width <= SINGLETRACK_MAX_WIDTH_M:
            return _NARROW
        return max(0.0, 1.0 - width / _WIDTH_DECAY_M)

    if highway in TRAIL_HIGHWAYS:
        if tags.mtb_scale is not None:
            return _TAGGED_TRAIL
        if surface_class in (SurfaceClass.COMPACTED, SurfaceClass.PAVED):
            # Often a surfaced walking path rather than a trail.
            return _ENGINEERED_TRAIL
        return _BARE_TRAIL

    if highway == "track":
        tracktype = first_semicolon_value(tags.tracktype)
        if tracktype in ROUGH_TRACKTYPES:
            return _ROUGH_TRACK
        if tracktype == "grade3":
            return _MIDDLING_TRACK

    return 0.0


def traffic_exposure(tags: WayTags) -> float:
    """How much motor traffic a rider shares the way with, from 0 to 1."""
    highway = first_semicolon_value(tags.highway)
    return TRAFFIC_EXPOSURE.get(highway or "", 0.0)
