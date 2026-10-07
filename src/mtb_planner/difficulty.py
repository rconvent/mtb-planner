"""Estimating how technical an edge is.

Everything normalises onto the `mtb:scale` 0–6 ordinal, because that is the
scale riders actually reason about and the only one with explicit data behind
it. Four levels of evidence, in descending order of trust:

* **Surveyed** — an explicit `mtb:scale` on this way. About 5% of trail ways.
* **Proxy** — `mtb:scale:imba` or `sac_scale`, which measure something close
  enough to convert.
* **Estimated** — a composite of surface, tracktype, smoothness and width.
  This is the overwhelming majority, and therefore the real model.
* **Default** — nothing but the way class was available.

The confidence travels with the number so nothing downstream can mistake an
estimate for a survey.
"""

from __future__ import annotations

from planner_core.features.enums import SurfaceClass, TechConfidence
from planner_core.ingest.tags import (
    MAX_MTB_SCALE,
    WayTags,
    first_semicolon_value,
    parse_imba_scale,
    parse_mtb_scale,
    parse_width_m,
)
from planner_core.sport import Difficulty

#: Starting technicality by way class, before any surface evidence.
HIGHWAY_BASE_TECH: dict[str, float] = {
    "path": 1.2,
    "bridleway": 1.0,
    "footway": 0.8,
    "track": 0.6,
    "steps": 3.0,
    "cycleway": 0.1,
    "service": 0.1,
    "pedestrian": 0.1,
    "residential": 0.0,
    "living_street": 0.0,
    "unclassified": 0.0,
    "road": 0.3,
}

#: How much each surface adds to estimated technicality.
SURFACE_TECH_DELTA: dict[SurfaceClass, float] = {
    SurfaceClass.PAVED: -0.6,
    SurfaceClass.COMPACTED: -0.3,
    SurfaceClass.GRAVEL: 0.0,
    SurfaceClass.GROUND: 0.4,
    SurfaceClass.GRASS: 0.4,
    SurfaceClass.ROCK: 1.0,
    SurfaceClass.SAND_MUD: 0.8,
    SurfaceClass.UNKNOWN: 0.0,
}

#: `smoothness` is a direct statement about rideability where it appears.
SMOOTHNESS_TECH_DELTA: dict[str, float] = {
    "excellent": -0.8,
    "good": -0.5,
    "intermediate": 0.0,
    "bad": 0.8,
    "very_bad": 1.5,
    "horrible": 2.3,
    "very_horrible": 3.0,
}

#: `tracktype` grades a track's firmness, and covers far more ways than
#: `surface` does on tracks specifically.
TRACKTYPE_TECH_DELTA: dict[str, float] = {
    "grade1": -0.5,
    "grade2": -0.2,
    "grade3": 0.2,
    "grade4": 0.6,
    "grade5": 1.0,
}

#: The Alpine hiking scale, converted onto the mountain-bike scale.
SAC_SCALE_TECH: dict[str, float] = {
    "strolling": 0.0,
    "hiking": 0.7,
    "mountain_hiking": 2.0,
    "demanding_mountain_hiking": 3.0,
    "alpine_hiking": 4.5,
    "demanding_alpine_hiking": 5.5,
    "difficult_alpine_hiking": 6.0,
}

#: IMBA grades trail-centre difficulty; the mapping is deliberately compressed
#: at the top, because an IMBA black is not an S4.
IMBA_TECH: dict[int, float] = {0: 0.5, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5}

#: How much narrowness adds to estimated technicality.
WIDTH_TECH_DELTA: tuple[tuple[float, float], ...] = (
    (0.6, 0.8),
    (1.0, 0.5),
    (1.5, 0.2),
    (2.5, 0.0),
)

#: Applied to widths above the last threshold above.
WIDE_TECH_DELTA = -0.3

#: `trail_visibility` values that imply route-finding difficulty.
TRAIL_VISIBILITY_TECH_DELTA: dict[str, float] = {
    "bad": 0.3,
    "horrible": 0.6,
    "no": 0.6,
}

#: An unofficial path is rougher than a maintained one.
INFORMAL_TECH_DELTA = 0.3

#: Uphill technicality, when untagged, is assumed a little below the downhill
#: figure: a rooty descent is awkward to climb but rarely as hard.
UPHILL_DISCOUNT = 0.5

#: `mtb:type` values implying purpose-built flow.
FLOW_MTB_TYPES: frozenset[str] = frozenset({"flowtrail", "flow", "singletrail"})

#: Confidence that an IMBA grade implies deliberate trail-centre design.
IMBA_FLOW = 0.7
FLOW_TYPE_SCORE = 1.0

#: Tags whose presence means the estimate rests on evidence, not defaults.
_EVIDENCE_FIELDS = ("surface", "tracktype", "smoothness", "width", "trail_visibility")


def _clamp(value: float) -> float:
    """Keep a technicality on the scale."""
    return min(max(value, 0.0), MAX_MTB_SCALE)


def _has_evidence(tags: WayTags) -> bool:
    """Whether any tag actually describes the running surface."""
    return any(getattr(tags, field) is not None for field in _EVIDENCE_FIELDS)


def _width_delta(tags: WayTags) -> float:
    """How much the way's width adds to technicality."""
    width = parse_width_m(tags.width)
    if width is None:
        return 0.0
    for threshold, delta in WIDTH_TECH_DELTA:
        if width <= threshold:
            return delta
    return WIDE_TECH_DELTA


def _flow_score(tags: WayTags) -> float:
    """How purpose-built and flowing an edge is likely to be."""
    mtb_type = first_semicolon_value(tags.mtb_type)
    if mtb_type in FLOW_MTB_TYPES:
        return FLOW_TYPE_SCORE
    if parse_imba_scale(tags.mtb_scale_imba) is not None:
        return IMBA_FLOW
    return 0.0


def _composite_tech(tags: WayTags, surface_class: SurfaceClass) -> float:
    """Estimate technicality from the tags that describe the surface.

    Surface and smoothness are combined with `max` rather than summed: they
    describe the same property from different angles, and adding them
    double-counts a way that is both rough and loose.
    """
    highway = first_semicolon_value(tags.highway)
    tech = HIGHWAY_BASE_TECH.get(highway or "", 0.0)

    surface_delta = SURFACE_TECH_DELTA[surface_class]
    smoothness = first_semicolon_value(tags.smoothness)
    smoothness_delta = SMOOTHNESS_TECH_DELTA.get(smoothness or "")
    tech += surface_delta if smoothness_delta is None else max(surface_delta, smoothness_delta)

    if highway == "track":
        tracktype = first_semicolon_value(tags.tracktype)
        tech += TRACKTYPE_TECH_DELTA.get(tracktype or "", 0.0)

    tech += _width_delta(tags)
    tech += TRAIL_VISIBILITY_TECH_DELTA.get(first_semicolon_value(tags.trail_visibility) or "", 0.0)
    if first_semicolon_value(tags.informal) == "yes":
        tech += INFORMAL_TECH_DELTA

    return _clamp(tech)


def _uphill_tech(tags: WayTags, downhill: float) -> float:
    """Technicality riding the other way."""
    explicit = parse_mtb_scale(tags.mtb_scale_uphill)
    if explicit is not None:
        return explicit
    return _clamp(downhill - UPHILL_DISCOUNT)


def estimate_difficulty(tags: WayTags, surface_class: SurfaceClass) -> Difficulty:
    """Derive an edge's technicality from the best evidence available."""
    flow = _flow_score(tags)

    surveyed = parse_mtb_scale(tags.mtb_scale)
    if surveyed is not None:
        return Difficulty(
            tech=surveyed,
            tech_up=_uphill_tech(tags, surveyed),
            confidence=TechConfidence.SURVEYED,
            flow=flow,
        )

    imba = parse_imba_scale(tags.mtb_scale_imba)
    if imba is not None:
        tech = IMBA_TECH[int(imba)]
        return Difficulty(
            tech=tech,
            tech_up=_uphill_tech(tags, tech),
            confidence=TechConfidence.PROXY,
            flow=max(flow, IMBA_FLOW),
        )

    sac = SAC_SCALE_TECH.get(first_semicolon_value(tags.sac_scale) or "")
    if sac is not None:
        return Difficulty(
            tech=sac,
            tech_up=_uphill_tech(tags, sac),
            confidence=TechConfidence.PROXY,
            flow=flow,
        )

    tech = _composite_tech(tags, surface_class)
    confidence = TechConfidence.ESTIMATED if _has_evidence(tags) else TechConfidence.DEFAULT
    return Difficulty(
        tech=tech,
        tech_up=_uphill_tech(tags, tech),
        confidence=confidence,
        flow=flow,
    )
