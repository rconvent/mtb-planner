"""Estimating how technical an edge is on foot.

The shared 0–6 ordinal is read as the Alpine hiking scale, which is what
`sac_scale` records and what runners who map trails actually use:

    0  smooth ground — road, towpath, gravel track
    1  T1  hiking: a clear path, flat or gently sloped
    2  T2  mountain hiking: a continuous trail, steep in places
    3  T3  demanding: exposed, sure-footedness needed
    4  T4  alpine hiking: hands sometimes needed
    5  T5  demanding alpine: scrambling
    6  T6  difficult alpine: climbing

Four levels of evidence, as on the bike, in descending order of trust:

* **Surveyed** — an explicit `sac_scale` on this way.
* **Proxy** — `mtb:scale`, which grades much the same ground for a different
  user. Converted with the top compressed: a rider's S4 is a walk-down for a
  runner more often than it is a T4.
* **Estimated** — a composite of surface, smoothness, trail visibility and
  width. On foot, visibility matters more than it does on a bike — losing
  the path costs a runner more than rough ground does — and narrowness and
  smoothness matter less.
* **Default** — nothing but the way class was available.
"""

from __future__ import annotations

from planner_core.features.enums import TECH_SCALE_MAX, SurfaceClass, TechConfidence
from planner_core.ingest.tags import WayTags, first_semicolon_value, parse_mtb_scale, parse_width_m
from planner_core.sport import Difficulty

#: `sac_scale` values onto the shared ordinal.
SAC_SCALE_TECH: dict[str, float] = {
    "strolling": 0.5,
    "hiking": 1.0,
    "mountain_hiking": 2.0,
    "demanding_mountain_hiking": 3.0,
    "alpine_hiking": 4.0,
    "demanding_alpine_hiking": 5.0,
    "difficult_alpine_hiking": 6.0,
}

#: `mtb:scale` grades onto the same ordinal, by whole grade.
MTB_SCALE_TECH: tuple[float, ...] = (0.8, 1.2, 2.0, 2.6, 3.2, 4.0, 4.5)

#: Starting technicality by way class, before any surface evidence.
HIGHWAY_BASE_TECH: dict[str, float] = {
    "path": 1.0,
    "bridleway": 1.0,
    "footway": 0.4,
    "track": 0.5,
    "steps": 1.2,
    "cycleway": 0.1,
    "service": 0.1,
    "pedestrian": 0.1,
    "residential": 0.0,
    "living_street": 0.0,
    "unclassified": 0.0,
    "road": 0.3,
}

#: How much each surface adds. Smaller than on a bike: loose gravel that
#: steals a rider's front wheel is merely noisy underfoot.
SURFACE_TECH_DELTA: dict[SurfaceClass, float] = {
    SurfaceClass.PAVED: -0.4,
    SurfaceClass.COMPACTED: -0.2,
    SurfaceClass.GRAVEL: 0.0,
    SurfaceClass.GROUND: 0.2,
    SurfaceClass.GRASS: 0.2,
    SurfaceClass.ROCK: 0.8,
    SurfaceClass.SAND_MUD: 0.6,
    SurfaceClass.UNKNOWN: 0.0,
}

#: `smoothness` grades wheels. It still says something about footing.
SMOOTHNESS_TECH_DELTA: dict[str, float] = {
    "excellent": -0.3,
    "good": -0.2,
    "intermediate": 0.0,
    "bad": 0.3,
    "very_bad": 0.6,
    "horrible": 1.0,
    "very_horrible": 1.4,
    "impassable": 1.8,
}

#: `tracktype` matters little on foot; a grade5 track is soft, not hard.
TRACKTYPE_TECH_DELTA: dict[str, float] = {
    "grade1": -0.3,
    "grade2": -0.1,
    "grade3": 0.1,
    "grade4": 0.3,
    "grade5": 0.5,
}

#: Route-finding is a runner's real difficulty on a faint path.
TRAIL_VISIBILITY_TECH_DELTA: dict[str, float] = {
    "intermediate": 0.2,
    "bad": 0.5,
    "horrible": 1.0,
    "no": 1.0,
}

#: Narrowness, by width threshold. A runner fits where a bike does not.
WIDTH_TECH_DELTA: tuple[tuple[float, float], ...] = ((0.6, 0.3), (1.0, 0.15))

#: An unofficial path is rougher than a maintained one.
INFORMAL_TECH_DELTA = 0.2

#: Technical ground is easier climbed than descended on foot, but less so than
#: on a bike: a runner has no momentum to lose on the way up.
UPHILL_DISCOUNT = 0.3

#: Tags whose presence means the estimate rests on evidence, not defaults.
_EVIDENCE_FIELDS = ("surface", "tracktype", "smoothness", "width", "trail_visibility")


def parse_sac_scale(value: str | None) -> float | None:
    """Parse a `sac_scale` value onto the shared 0–6 ordinal."""
    if value is None:
        return None
    return SAC_SCALE_TECH.get(value.strip().lower())


def _clamp(value: float) -> float:
    """Keep a technicality on the scale."""
    return min(max(value, 0.0), TECH_SCALE_MAX)


def _has_evidence(tags: WayTags) -> bool:
    """Whether any tag actually describes the ground underfoot."""
    return any(getattr(tags, field) is not None for field in _EVIDENCE_FIELDS)


def _width_delta(tags: WayTags) -> float:
    """How much the way's width adds to technicality."""
    width = parse_width_m(tags.width)
    if width is None:
        return 0.0
    for threshold, delta in WIDTH_TECH_DELTA:
        if width <= threshold:
            return delta
    return 0.0


def _composite_tech(tags: WayTags, surface_class: SurfaceClass) -> float:
    """Estimate technicality from the tags that describe the ground.

    Surface and smoothness describe the same property from different angles,
    so the larger is taken rather than the sum.
    """
    highway = first_semicolon_value(tags.highway)
    tech = HIGHWAY_BASE_TECH.get(highway or "", 0.0)

    surface_delta = SURFACE_TECH_DELTA[surface_class]
    smoothness_delta = SMOOTHNESS_TECH_DELTA.get(first_semicolon_value(tags.smoothness) or "")
    tech += surface_delta if smoothness_delta is None else max(surface_delta, smoothness_delta)

    if highway == "track":
        tech += TRACKTYPE_TECH_DELTA.get(first_semicolon_value(tags.tracktype) or "", 0.0)

    tech += _width_delta(tags)
    tech += TRAIL_VISIBILITY_TECH_DELTA.get(first_semicolon_value(tags.trail_visibility) or "", 0.0)
    if first_semicolon_value(tags.informal) == "yes":
        tech += INFORMAL_TECH_DELTA

    return _clamp(tech)


def _difficulty(tech: float, confidence: TechConfidence) -> Difficulty:
    """Package a technicality, deriving the uphill figure from it."""
    return Difficulty(
        tech=tech,
        tech_up=_clamp(tech - UPHILL_DISCOUNT),
        confidence=confidence,
        # Flow is a property of trail built for wheels. A runner's profiles do
        # not weight it, and leaving it at zero keeps that honest.
        flow=0.0,
    )


def estimate_difficulty(tags: WayTags, surface_class: SurfaceClass) -> Difficulty:
    """Derive an edge's technicality on foot from the best evidence available."""
    surveyed = parse_sac_scale(tags.sac_scale)
    if surveyed is not None:
        return _difficulty(surveyed, TechConfidence.SURVEYED)

    mtb = parse_mtb_scale(tags.mtb_scale)
    if mtb is not None:
        return _difficulty(MTB_SCALE_TECH[round(mtb)], TechConfidence.PROXY)

    tech = _composite_tech(tags, surface_class)
    confidence = TechConfidence.ESTIMATED if _has_evidence(tags) else TechConfidence.DEFAULT
    return _difficulty(tech, confidence)
