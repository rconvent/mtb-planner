"""A completed route and its measurements.

Distance is summed from edge lengths rather than re-measured from the exported
polyline, so what the rider is told matches what the search actually costed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mtb_planner.core.geo import LatLon
from mtb_planner.graph.store import GraphStore
from mtb_planner.ingest.way_filter import BikeAccess


@dataclass(frozen=True)
class RouteStats:
    """Measurements of a route."""

    distance_m: float
    n_edges: int
    n_points: int
    dismount_m: float
    ascent_m: float = 0.0
    descent_m: float = 0.0
    moving_time_s: float = 0.0

    @property
    def distance_km(self) -> float:
        """Route length in kilometres."""
        return self.distance_m / 1000.0

    @property
    def moving_time_min(self) -> float:
        """Estimated riding time in minutes."""
        return self.moving_time_s / 60.0

    @property
    def average_speed_km_h(self) -> float:
        """Distance over estimated time, in km/h."""
        if self.moving_time_s <= 0.0:
            return 0.0
        return (self.distance_m / 1000.0) / (self.moving_time_s / 3600.0)


@dataclass(frozen=True)
class Route:
    """A traversable sequence of arcs, with geometry and measurements."""

    arcs: tuple[int, ...]
    nodes: tuple[int, ...]
    geometry: npt.NDArray[np.float64]
    stats: RouteStats
    vertex_index: tuple[int, ...] = ()
    """Index in `geometry` of the node each arc starts from, and finally ends at.

    Length is `len(arcs) + 1`, so arc `i` covers
    `geometry[vertex_index[i] : vertex_index[i + 1] + 1]` — inclusive of both
    boundaries, because consecutive arcs share the junction between them.

    Recorded here, derived from the chunks the geometry was actually built from,
    so the API can describe each segment's difficulty by index instead of
    shipping a second copy of the polyline alongside it.
    """

    @property
    def start(self) -> LatLon:
        """First point of the route."""
        return LatLon(lat=float(self.geometry[0, 0]), lon=float(self.geometry[0, 1]))

    @property
    def end(self) -> LatLon:
        """Last point of the route."""
        return LatLon(lat=float(self.geometry[-1, 0]), lon=float(self.geometry[-1, 1]))

    @property
    def is_loop(self) -> bool:
        """Whether the route returns to where it started."""
        return len(self.nodes) > 1 and self.nodes[0] == self.nodes[-1]


def build_route(
    store: GraphStore,
    source: int,
    arcs: Sequence[int],
    arc_time_s: npt.NDArray[np.float64] | None = None,
) -> Route:
    """Assemble a `Route` from a search's arc sequence.

    Consecutive arcs share a junction, so each arc after the first contributes
    its geometry from the second point onward; otherwise every junction would
    appear twice in the exported track.
    """
    if not arcs:
        point = store.node_point(source)
        geometry = np.array([[point.lat, point.lon]], dtype=np.float64)
        return Route(
            arcs=(),
            nodes=(source,),
            geometry=geometry,
            stats=RouteStats(distance_m=0.0, n_edges=0, n_points=1, dismount_m=0.0),
            vertex_index=(0,),
        )

    chunks: list[npt.NDArray[np.float64]] = []
    nodes: list[int] = [source]
    distance = 0.0
    dismount = 0.0
    ascent = 0.0
    descent = 0.0

    for position, arc in enumerate(arcs):
        points = store.arc_geometry(arc)
        chunks.append(points if position == 0 else points[1:])
        nodes.append(int(store.arc_head[arc]))
        edge = int(store.arc_edge[arc])
        length = float(store.edge_length_m[edge])
        distance += length
        if store.edge_access[edge] == BikeAccess.DISMOUNT:
            dismount += length

        # Elevation is stored along the edge's own direction, so an arc
        # running against it climbs what the edge descends.
        edge_up = float(store.elevation.ascent_m[edge])
        edge_down = float(store.elevation.descent_m[edge])
        if store.arc_forward[arc]:
            ascent += edge_up
            descent += edge_down
        else:
            ascent += edge_down
            descent += edge_up

    geometry = np.vstack(chunks)

    # Boundaries from the chunk lengths themselves, so they cannot disagree with
    # the geometry they describe. Arc 0 contributed all its points and so spans
    # one fewer boundary than it has points; every arc after it dropped the
    # junction it shares with its predecessor, so its chunk length *is* its
    # advance.
    boundaries = [0]
    for position, chunk in enumerate(chunks):
        advance = int(chunk.shape[0]) - (1 if position == 0 else 0)
        boundaries.append(boundaries[-1] + advance)

    return Route(
        arcs=tuple(int(a) for a in arcs),
        nodes=tuple(nodes),
        geometry=geometry,
        vertex_index=tuple(boundaries),
        stats=RouteStats(
            distance_m=distance,
            n_edges=len(arcs),
            n_points=int(geometry.shape[0]),
            dismount_m=dismount,
            ascent_m=ascent,
            descent_m=descent,
            # Summed from the arcs actually ridden, never derived from the
            # cost that drove the search — that is what keeps it honest.
            moving_time_s=(0.0 if arc_time_s is None else float(sum(arc_time_s[a] for a in arcs))),
        ),
    )
