"""Categorical edge features.

Small integer enums rather than strings: these end up as `uint8` columns over
millions of edges, and comparing integers is what makes the per-request cost
vector a vectorised operation instead of a Python loop.
"""

from __future__ import annotations

from enum import IntEnum


class TechConfidence(IntEnum):
    """How much is actually known about an edge's technicality.

    Surfaced through the API so the interface can say "estimated" where it is
    estimating. At 5% explicit coverage, presenting inference as survey would
    be dishonest.
    """

    #: Nothing but the way class was available.
    DEFAULT = 0
    #: Inferred from surface, tracktype, smoothness and width.
    ESTIMATED = 1
    #: From a related scale, or propagated from surveyed neighbours.
    PROXY = 2
    #: An explicit `mtb:scale` on this way.
    SURVEYED = 3


class SurfaceClass(IntEnum):
    """What the wheels are actually rolling on.

    Ordered roughly by rolling resistance, which is how the speed model will
    read it.
    """

    PAVED = 0
    COMPACTED = 1
    GRAVEL = 2
    GROUND = 3
    GRASS = 4
    ROCK = 5
    SAND_MUD = 6
    UNKNOWN = 7


class WayClass(IntEnum):
    """The shape of the way, as a rider would describe it."""

    SINGLETRACK = 0
    DOUBLETRACK = 1
    FIREROAD = 2
    GRAVEL_ROAD = 3
    PAVED_MINOR = 4
    PAVED_MAJOR = 5
    URBAN = 6
    STEPS = 7
    FERRY = 8
    UNKNOWN = 9


#: Surfaces that are sealed. Kept separate from the enum ordering so adding a
#: class later cannot silently change what counts as paved.
PAVED_SURFACES: frozenset[SurfaceClass] = frozenset({SurfaceClass.PAVED})

#: Way classes with no unsealed character at all.
PAVED_WAY_CLASSES: frozenset[WayClass] = frozenset(
    {WayClass.PAVED_MINOR, WayClass.PAVED_MAJOR, WayClass.URBAN}
)
