"""Which ways are rideable, and in which direction.

Two decisions live here, and both have bitten this kind of project before:

* `access=forestry` is **not** a bicycle ban. Wallonian forest tracks carry it
  constantly; excluding them deletes most of the Ardennes network.
* `oneway:bicycle=no` alongside `oneway=yes` is standard Belgian contraflow
  tagging (26,594 ways). Ignoring it produces absurd urban detours.

Legal exclusions are hard, never soft penalties. A route through a way tagged
`bicycle=no` is not a slightly worse route; it is a wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from mtb_planner.ingest.tags import WayTags, first_semicolon_value


class BikeAccess(IntEnum):
    """Whether, and how, a way may be used on a mountain bike."""

    NO = 0
    DISMOUNT = 1
    YES = 2


class Direction(IntEnum):
    """Permitted directions of travel along a way."""

    BOTH = 0
    FORWARD = 1
    BACKWARD = 2


@dataclass(frozen=True, slots=True)
class OnewayRule:
    """A resolved direction, and whether the evidence for it is trustworthy.

    A stray `oneway=yes` on a `highway=path` is a common mapping error, and
    enforcing it can silently make a whole area unloopable. Restrictions on
    paths are therefore only treated as hard when corroborated by a tag that
    implies deliberate directional design.
    """

    direction: Direction
    corroborated: bool


#: Rideable way classes. `footway` is included because OSM uses it for forest
#: paths as well as urban pavements; the scoring model separates them later.
INCLUDED_HIGHWAYS: frozenset[str] = frozenset(
    {
        "path",
        "track",
        "cycleway",
        "bridleway",
        "footway",
        "steps",
        "living_street",
        "residential",
        "unclassified",
        "service",
        "tertiary",
        "tertiary_link",
        "secondary",
        "secondary_link",
        "primary",
        "primary_link",
        "road",
        "pedestrian",
    }
)

#: Bicycles are banned outright, or the way is not a way at all.
EXCLUDED_HIGHWAYS: frozenset[str] = frozenset(
    {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "construction",
        "proposed",
        "abandoned",
        "razed",
        "raceway",
        "bus_guideway",
        "busway",
        "platform",
        "corridor",
        "elevator",
        "rest_area",
        "services",
    }
)

#: Private access ways that only serve a building or car park. Excluding these
#: removes a very large share of `highway=service` without losing anything
#: rideable, unless the way carries an explicit bicycle tag.
EXCLUDED_SERVICE: frozenset[str] = frozenset({"driveway", "parking_aisle", "drive-through"})

#: Values that grant bicycle access, overriding a restrictive `access` tag.
POSITIVE_BICYCLE: frozenset[str] = frozenset(
    {"yes", "designated", "permissive", "destination", "official", "public"}
)

#: Values on `access` that block the public. Deliberately excludes `forestry`
#: and `agricultural`, which restrict motor traffic rather than bicycles.
BLOCKING_ACCESS: frozenset[str] = frozenset({"private", "no"})

#: Way classes where a `oneway` tag is often a mapping slip rather than a rule.
PATH_LIKE_HIGHWAYS: frozenset[str] = frozenset(
    {"path", "track", "cycleway", "footway", "bridleway", "steps"}
)

_ONEWAY_FORWARD: frozenset[str] = frozenset({"yes", "true", "1"})
_ONEWAY_BACKWARD: frozenset[str] = frozenset({"-1", "reverse"})
_ONEWAY_NONE: frozenset[str] = frozenset({"no", "false", "0"})


def _normalise(value: str | None) -> str | None:
    """Lower-case a tag value and take the head of any semicolon list."""
    return first_semicolon_value(value)


def resolve_bike_access(tags: WayTags) -> BikeAccess:
    """Decide whether a way may be ridden, pushed, or not used at all."""
    highway = _normalise(tags.highway)

    # Ferries have no highway tag but do carry bikes across the Meuse.
    if highway is None:
        return BikeAccess.YES if _normalise(tags.route) == "ferry" else BikeAccess.NO

    if _normalise(tags.area) == "yes":
        return BikeAccess.NO
    if highway in EXCLUDED_HIGHWAYS or highway not in INCLUDED_HIGHWAYS:
        return BikeAccess.NO
    if _normalise(tags.smoothness) == "impassable":
        return BikeAccess.NO

    bicycle = _normalise(tags.bicycle)

    # Explicit prohibition. Hard, never a penalty — this is a legal boundary.
    if bicycle == "no":
        return BikeAccess.NO

    # A private or closed way stays closed unless bicycles are named.
    if _normalise(tags.access) in BLOCKING_ACCESS and bicycle not in POSITIVE_BICYCLE:
        return BikeAccess.NO

    # Service roads that only reach a driveway or car park, with no bike tag.
    if highway == "service" and _normalise(tags.service) in EXCLUDED_SERVICE:
        return BikeAccess.YES if bicycle in POSITIVE_BICYCLE else BikeAccess.NO

    # Rideable in principle, but not actually ridden.
    if bicycle == "dismount" or highway == "steps":
        return BikeAccess.DISMOUNT

    return BikeAccess.YES


def is_bike_relevant(tags: WayTags) -> bool:
    """Whether a way should enter the routing graph at all."""
    return resolve_bike_access(tags) is not BikeAccess.NO


def _has_directional_evidence(tags: WayTags) -> bool:
    """Whether a path's `oneway` tag looks deliberate rather than accidental.

    Descent-only trails are tagged by people who ride them, and those people
    also tend to record technicality or an explicit downhill incline.
    """
    return any(
        (
            tags.mtb_scale is not None,
            tags.mtb_type is not None,
            tags.mtb_scale_imba is not None,
            _normalise(tags.incline) == "down",
        )
    )


def _direction_from_value(value: str | None) -> Direction | None:
    """Map a raw `oneway`-style value onto a direction, if it is meaningful."""
    normalised = _normalise(value)
    if normalised is None:
        return None
    if normalised in _ONEWAY_FORWARD:
        return Direction.FORWARD
    if normalised in _ONEWAY_BACKWARD:
        return Direction.BACKWARD
    if normalised in _ONEWAY_NONE:
        return Direction.BOTH
    return None


def _has_contraflow_cycleway(tags: WayTags) -> bool:
    """Whether a contraflow cycle lane makes a one-way street bidirectional."""
    return any(
        (value or "").lower().startswith("opposite")
        for value in (tags.cycleway, tags.cycleway_left, tags.cycleway_right)
    )


def resolve_direction(tags: WayTags) -> OnewayRule:
    """Decide which directions of a way a bicycle may travel.

    Bicycle-specific tags win over generic ones, because a street can be
    one-way for cars and two-way for bikes — which in Belgium it very often is.
    """
    # 1. Mountain-bike specific, then bicycle specific. Both are deliberate.
    for value in (tags.oneway_mtb, tags.oneway_bicycle):
        if (direction := _direction_from_value(value)) is not None:
            return OnewayRule(direction, corroborated=True)

    # 2. A contraflow lane means bikes may go both ways regardless of `oneway`.
    if _has_contraflow_cycleway(tags):
        return OnewayRule(Direction.BOTH, corroborated=True)

    # 3. Roundabouts are directional even when `oneway` is absent.
    if _normalise(tags.junction) == "roundabout":
        return OnewayRule(Direction.FORWARD, corroborated=True)

    # 4. The generic tag. Reliable on roads, suspect on paths.
    if (direction := _direction_from_value(tags.oneway)) is not None:
        if direction is Direction.BOTH:
            return OnewayRule(Direction.BOTH, corroborated=True)
        highway = _normalise(tags.highway)
        if highway in PATH_LIKE_HIGHWAYS:
            return OnewayRule(direction, corroborated=_has_directional_evidence(tags))
        return OnewayRule(direction, corroborated=True)

    return OnewayRule(Direction.BOTH, corroborated=True)
