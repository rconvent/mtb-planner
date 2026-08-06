"""Surface and smoothness.

`surface` appears on 94,033 Luxembourg ways and `tracktype` on 34,242, against
2,709 carrying `mtb:scale`. These are the tags the difficulty model actually
runs on, so their handling matters more than the explicit scale does.

Where `surface` is absent the class is guessed from the way type. Those
defaults are regional: a Belgian `highway=track` without `tracktype` is
forestry gravel in the Ardennes but often a concrete farm track in Flanders.
"""

from __future__ import annotations

from mtb_planner.features.enums import SurfaceClass
from mtb_planner.ingest.tags import WayTags, first_semicolon_value

#: OSM `surface` values, mapped to the class the speed model understands.
SURFACE_CLASSES: dict[str, SurfaceClass] = {
    # Sealed.
    "asphalt": SurfaceClass.PAVED,
    "concrete": SurfaceClass.PAVED,
    "concrete:plates": SurfaceClass.PAVED,
    "concrete:lanes": SurfaceClass.PAVED,
    "paved": SurfaceClass.PAVED,
    "chipseal": SurfaceClass.PAVED,
    "metal": SurfaceClass.PAVED,
    "wood": SurfaceClass.PAVED,
    "paving_stones": SurfaceClass.PAVED,
    "sett": SurfaceClass.PAVED,
    "cobblestone": SurfaceClass.ROCK,
    "unhewn_cobblestone": SurfaceClass.ROCK,
    # Firm unsealed.
    "compacted": SurfaceClass.COMPACTED,
    "fine_gravel": SurfaceClass.COMPACTED,
    # Loose unsealed.
    "gravel": SurfaceClass.GRAVEL,
    "pebblestone": SurfaceClass.GRAVEL,
    "unpaved": SurfaceClass.GRAVEL,
    # Natural.
    "ground": SurfaceClass.GROUND,
    "dirt": SurfaceClass.GROUND,
    "earth": SurfaceClass.GROUND,
    "soil": SurfaceClass.GROUND,
    "woodchips": SurfaceClass.GROUND,
    "grass": SurfaceClass.GRASS,
    "grass_paver": SurfaceClass.GRASS,
    "rock": SurfaceClass.ROCK,
    "stone": SurfaceClass.ROCK,
    "sand": SurfaceClass.SAND_MUD,
    "mud": SurfaceClass.SAND_MUD,
}

#: What to assume when `surface` is absent, by way class.
DEFAULT_SURFACE_BY_HIGHWAY: dict[str, SurfaceClass] = {
    "path": SurfaceClass.GROUND,
    "bridleway": SurfaceClass.GROUND,
    "track": SurfaceClass.COMPACTED,
    "footway": SurfaceClass.PAVED,
    "steps": SurfaceClass.PAVED,
    "cycleway": SurfaceClass.PAVED,
    "residential": SurfaceClass.PAVED,
    "living_street": SurfaceClass.PAVED,
    "unclassified": SurfaceClass.PAVED,
    "tertiary": SurfaceClass.PAVED,
    "tertiary_link": SurfaceClass.PAVED,
    "secondary": SurfaceClass.PAVED,
    "secondary_link": SurfaceClass.PAVED,
    "primary": SurfaceClass.PAVED,
    "primary_link": SurfaceClass.PAVED,
    "service": SurfaceClass.PAVED,
    "pedestrian": SurfaceClass.PAVED,
    "road": SurfaceClass.COMPACTED,
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

#: How well each `smoothness` value rolls, from 0 (barely) to 1 (perfectly).
SMOOTHNESS_ROLL: dict[str, float] = {
    "excellent": 1.0,
    "good": 0.85,
    "intermediate": 0.65,
    "bad": 0.45,
    "very_bad": 0.3,
    "horrible": 0.15,
    "very_horrible": 0.05,
}

#: Fallback rollability per surface, used when `smoothness` is absent.
SURFACE_ROLL: dict[SurfaceClass, float] = {
    SurfaceClass.PAVED: 1.0,
    SurfaceClass.COMPACTED: 0.8,
    SurfaceClass.GRAVEL: 0.65,
    SurfaceClass.GROUND: 0.55,
    SurfaceClass.GRASS: 0.4,
    SurfaceClass.ROCK: 0.3,
    SurfaceClass.SAND_MUD: 0.2,
    SurfaceClass.UNKNOWN: 0.6,
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


def classify_surface(tags: WayTags) -> SurfaceClass:
    """Decide what an edge is surfaced with.

    Falls back to a per-way-class default rather than `UNKNOWN`, because
    "unknown" propagates as a neutral value and would flatten the difference
    between a forest track and a residential street.
    """
    surface = first_semicolon_value(tags.surface)
    if surface is not None and surface in SURFACE_CLASSES:
        return SURFACE_CLASSES[surface]

    highway = first_semicolon_value(tags.highway)
    if highway is None:
        return SurfaceClass.UNKNOWN
    return DEFAULT_SURFACE_BY_HIGHWAY.get(highway, SurfaceClass.UNKNOWN)


def surface_is_explicit(tags: WayTags) -> bool:
    """Whether the surface was tagged rather than assumed."""
    surface = first_semicolon_value(tags.surface)
    return surface is not None and surface in SURFACE_CLASSES


def rollability(tags: WayTags, surface_class: SurfaceClass) -> float:
    """How well an edge rolls, from 0 (barely) to 1 (perfectly).

    `smoothness` wins where present: it is a rider's direct judgement, while
    surface is only a material.
    """
    smoothness = first_semicolon_value(tags.smoothness)
    if smoothness is not None and smoothness in SMOOTHNESS_ROLL:
        return SMOOTHNESS_ROLL[smoothness]
    return SURFACE_ROLL[surface_class]
