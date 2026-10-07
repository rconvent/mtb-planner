"""Which ways a runner may use, and in which direction.

On foot the legal picture is far simpler than on a bike, and the differences
are exactly where the two planners must disagree:

* `bicycle=no` says nothing about runners. The footpaths a rider may not take
  are the ones a runner most wants.
* A one-way mountain-bike descent, or a one-way street, binds traffic on
  wheels. Only `oneway:foot` binds a runner.
* `smoothness=impassable` means no wheeled vehicle gets through. Feet do.
* Steps are simply part of the run.

Legal exclusions are still hard, never soft penalties: `foot=no`, and a
private or closed way that does not name pedestrians, are removed from the
graph. `foot=use_sidepath` is a closure too — it sends runners to the parallel
footway, which is in the graph in its own right.
"""

from __future__ import annotations

from planner_core.ingest.oneway import direction_from_value
from planner_core.ingest.tags import WayTags, first_semicolon_value
from planner_core.sport import Access, Direction, OnewayRule

#: Ways a runner may use. Roads are included because they are how trails
#: connect; the cost model, not the filter, keeps a runner off them.
INCLUDED_HIGHWAYS: frozenset[str] = frozenset(
    {
        "path",
        "track",
        "footway",
        "steps",
        "bridleway",
        "pedestrian",
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
        # RAVeL greenways and many forest paths are tagged as cycleways and
        # walked as much as ridden. An explicit `foot=no` still closes one.
        "cycleway",
    }
)

#: Values on `foot` that close a way to pedestrians.
BLOCKING_FOOT: frozenset[str] = frozenset({"no", "use_sidepath"})

#: Values that grant pedestrian access, overriding a restrictive `access` tag.
POSITIVE_FOOT: frozenset[str] = frozenset(
    {"yes", "designated", "permissive", "destination", "official", "public"}
)

#: Values on `access` that block the public. As for bikes, `forestry` and
#: `agricultural` restrict motor traffic, not people on foot.
BLOCKING_ACCESS: frozenset[str] = frozenset({"private", "no"})

#: Private access ways that only serve a building or car park.
EXCLUDED_SERVICE: frozenset[str] = frozenset({"driveway", "parking_aisle", "drive-through"})

#: Hiking grades that need hands more than legs. Passable, but at a scramble,
#: so they are kept and timed at walking pace instead of a running speed.
SCRAMBLE_SAC_SCALES: frozenset[str] = frozenset(
    {"demanding_alpine_hiking", "difficult_alpine_hiking"}
)


def resolve_access(tags: WayTags) -> Access:
    """Decide whether a way may be run, only scrambled, or not used at all."""
    highway = first_semicolon_value(tags.highway)

    # Foot passengers are as welcome on a ferry as bikes are.
    if highway is None:
        return Access.YES if first_semicolon_value(tags.route) == "ferry" else Access.NO

    if first_semicolon_value(tags.area) == "yes":
        return Access.NO
    if highway not in INCLUDED_HIGHWAYS:
        return Access.NO

    foot = first_semicolon_value(tags.foot)

    # Explicit prohibition. Hard, never a penalty — this is a legal boundary.
    if foot in BLOCKING_FOOT:
        return Access.NO

    # A private or closed way stays closed unless pedestrians are named.
    if first_semicolon_value(tags.access) in BLOCKING_ACCESS and foot not in POSITIVE_FOOT:
        return Access.NO

    # Service roads that only reach a driveway or car park.
    if highway == "service" and first_semicolon_value(tags.service) in EXCLUDED_SERVICE:
        return Access.YES if foot in POSITIVE_FOOT else Access.NO

    if first_semicolon_value(tags.sac_scale) in SCRAMBLE_SAC_SCALES:
        return Access.WALK

    return Access.YES


def resolve_direction(tags: WayTags) -> OnewayRule:
    """Decide which directions of a way a runner may travel.

    Only `oneway:foot` applies. `oneway` on a street is for vehicles, and a
    one-way trail is one-way for the bikes descending it — a runner climbing it
    is legal, if worth keeping an eye out on.
    """
    direction = direction_from_value(tags.oneway_foot)
    return OnewayRule(direction or Direction.BOTH, corroborated=True)
