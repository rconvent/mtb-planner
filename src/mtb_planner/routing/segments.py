"""Breaking a route into the pieces a map and a chart need.

The route the router returns is one polyline and a set of totals. A map that
colours difficulty along the line, dashes the parts where difficulty was
inferred rather than surveyed, and links a hovered point on an elevation chart
back to a position on the trail needs rather more: it needs to know, per edge,
where that edge sits along the route and what is known about it.

All of it comes from data already in the artifact. Two consequences are worth
stating plainly:

**The elevation profile is relative, not altitude.** The artifact stores per-edge
ascent and descent — the smoothed, dead-banded figures — but no absolute height,
so there is nothing to anchor a profile to sea level. What is recoverable is
height *change from the start*, cumulated edge by edge, which is what an
elevation chart actually draws. It is consistent with the reported total ascent
by construction, because it is summed from the same numbers.

**Its resolution is one point per edge, not per vertex.** The wiggle inside an
edge is not recorded, only its net rise. On a 25 km route that is still a few
hundred points, which is more than a chart a few hundred pixels wide can show.

Segments describe themselves by index into the route's geometry rather than
carrying their own copy of it, so adding difficulty to a response costs a few
numbers per edge instead of a second polyline.
"""

from __future__ import annotations

from dataclasses import dataclass

from mtb_planner.features.enums import TechConfidence
from mtb_planner.graph.store import GraphStore
from mtb_planner.routing.result import Route


@dataclass(frozen=True)
class RouteSegment:
    """One edge of a route, as the front end needs it."""

    first: int
    """Index in the route geometry of this segment's first point."""

    last: int
    """Index of its last point. Shared with the next segment's `first`."""

    at_m: float
    """Distance along the route where this segment begins."""

    length_m: float

    tech: float
    """Technicality in the direction ridden, on the `mtb:scale` 0 to 6 ordinal."""

    surveyed: bool
    """Whether that difficulty was mapped rather than inferred.

    Drives a distinct dash pattern on the map. At a few percent survey coverage,
    drawing inferred difficulty exactly like surveyed difficulty would be a
    claim the data does not support.
    """

    rise_m: float
    """Net height change along this segment, signed."""


@dataclass(frozen=True)
class ElevationPoint:
    """Height relative to the start, at a distance along the route."""

    at_m: float
    height_m: float


def route_segments(store: GraphStore, route: Route) -> tuple[RouteSegment, ...]:
    """Describe every edge of a route in place along it."""
    if not route.arcs or len(route.vertex_index) != len(route.arcs) + 1:
        return ()

    features = store.features
    segments: list[RouteSegment] = []
    travelled = 0.0

    for position, arc in enumerate(route.arcs):
        edge = int(store.arc_edge[arc])
        length = float(store.edge_length_m[edge])
        forward = bool(store.arc_forward[arc])

        # Difficulty is directional: a rooty descent is a different proposition
        # climbed, and the artifact records both.
        tech = float(features.tech[edge] if forward else features.tech_up[edge])

        up = float(store.elevation.ascent_m[edge])
        down = float(store.elevation.descent_m[edge])
        rise = (up - down) if forward else (down - up)

        segments.append(
            RouteSegment(
                first=route.vertex_index[position],
                last=route.vertex_index[position + 1],
                at_m=travelled,
                length_m=length,
                tech=tech,
                surveyed=int(features.tech_conf[edge]) >= int(TechConfidence.SURVEYED),
                rise_m=rise,
            )
        )
        travelled += length

    return tuple(segments)


def elevation_profile(segments: tuple[RouteSegment, ...]) -> tuple[ElevationPoint, ...]:
    """Height relative to the start, at every segment boundary.

    Starts at zero by definition — there is no absolute height to start from —
    so the chart shows the shape of the climbing rather than an altitude.
    """
    if not segments:
        return ()

    points = [ElevationPoint(at_m=0.0, height_m=0.0)]
    height = 0.0
    for segment in segments:
        height += segment.rise_m
        points.append(ElevationPoint(at_m=segment.at_m + segment.length_m, height_m=height))
    return tuple(points)
