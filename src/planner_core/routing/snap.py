"""Turning a clicked coordinate into a node the search can start from.

Two failures are distinguished, because they need different answers from the
rider. Outside the built area means "this planner does not cover that place".
Off-network means "there is no trail there, but there is one 300 m away" — and
the second is only useful if the nearest point comes back with it.
"""

from __future__ import annotations

from dataclasses import dataclass

from planner_core.core.errors import PointOffNetworkError, PointOutsideCoverageError
from planner_core.core.geo import BBox, LatLon, haversine_m
from planner_core.graph.spatial_index import SpatialIndex
from planner_core.graph.store import GraphStore


@dataclass(frozen=True)
class Snap:
    """Where a requested point was attached to the network."""

    node: int
    edge: int
    distance_m: float
    point: LatLon


def snap_point(
    store: GraphStore,
    index: SpatialIndex,
    point: LatLon,
    max_distance_m: float,
    coverage: BBox | None = None,
) -> Snap:
    """Attach a coordinate to the nearest routable node.

    The nearest *way* determines whether the point is on-network at all; the
    node the search starts from is then an endpoint of that way, provided it
    is in the main component. Falling back to the nearest routable node keeps
    a request usable when the closest trail turns out to be an island.
    """
    if coverage is not None and not coverage.contains(point):
        raise PointOutsideCoverageError(lat=point.lat, lon=point.lon)

    edge, distance_m, snapped = index.nearest_edge(point)
    if distance_m > max_distance_m:
        raise PointOffNetworkError(
            lat=point.lat,
            lon=point.lon,
            nearest_lat=snapped.lat,
            nearest_lon=snapped.lon,
            distance_m=distance_m,
        )

    endpoints = [int(store.edge_u[edge]), int(store.edge_v[edge])]
    routable = [node for node in endpoints if store.in_giant_component(node)]

    if routable:
        node = min(routable, key=lambda n: haversine_m(store.node_point(n), snapped))
    else:
        node, _ = index.nearest_routable_node(point)

    return Snap(node=node, edge=edge, distance_m=distance_m, point=snapped)
