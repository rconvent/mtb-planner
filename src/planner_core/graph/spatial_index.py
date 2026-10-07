"""Nearest-way and nearest-node lookup.

Indexes every *geometry* point, not just junctions, so "how far is the nearest
trail" is answered against the actual line rather than its endpoints. That
distinction only grows in importance: once edges are contracted, endpoints can
be hundreds of metres from the point a rider clicked.

Uses `scipy.spatial.cKDTree` rather than `rtree`, which has no wheels for
current Python on Windows and would mean a source build.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from planner_core.core.geo import LatLon
from planner_core.core.units import EARTH_RADIUS_M
from planner_core.graph.store import GraphStore

#: Metres per degree of latitude. Longitude is scaled by cos(latitude) about
#: the graph's centre, which is accurate to well under a metre across an area
#: the size of Belgium — ample for nearest-neighbour ranking.
_M_PER_DEG = EARTH_RADIUS_M * math.pi / 180.0


def _project(
    lat: npt.NDArray[np.float64], lon: npt.NDArray[np.float64], lat0: float
) -> npt.NDArray[np.float64]:
    """Equirectangular projection to local metres about `lat0`."""
    return np.column_stack((lat * _M_PER_DEG, lon * _M_PER_DEG * math.cos(math.radians(lat0))))


class SpatialIndex:
    """Nearest-edge and nearest-routable-node queries over a graph."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store
        self._lat0 = float(store.node_lat.mean())

        # Every geometry point, mapped back to the edge it belongs to.
        counts = np.diff(store.edge_geom_offset)
        self._point_edge = np.repeat(
            np.arange(store.n_edges, dtype=np.int32), counts.astype(np.int64)
        )
        self._edge_tree = cKDTree(_project(store.geom[:, 0], store.geom[:, 1], self._lat0))

        # Routable nodes only: snapping into an isolated fragment guarantees
        # a "no route found" that the rider cannot act on.
        self._routable = np.flatnonzero(store.node_scc == store.giant_scc).astype(np.int32)
        self._node_tree = cKDTree(
            _project(store.node_lat[self._routable], store.node_lon[self._routable], self._lat0)
        )

    @property
    def n_routable_nodes(self) -> int:
        """Number of nodes reachable within the main network."""
        return int(self._routable.shape[0])

    def _query_point(self, point: LatLon) -> npt.NDArray[np.float64]:
        return _project(
            np.array([point.lat], dtype=np.float64),
            np.array([point.lon], dtype=np.float64),
            self._lat0,
        )

    def nearest_edge(self, point: LatLon) -> tuple[int, float, LatLon]:
        """Return the nearest edge, its distance in metres, and the point on it."""
        distance, index = self._edge_tree.query(self._query_point(point), k=1)
        point_index = int(index[0])
        edge = int(self._point_edge[point_index])
        snapped = self._store.geom[point_index]
        return edge, float(distance[0]), LatLon(lat=float(snapped[0]), lon=float(snapped[1]))

    def nearest_routable_node(self, point: LatLon) -> tuple[int, float]:
        """Return the nearest node in the main component, and its distance."""
        distance, index = self._node_tree.query(self._query_point(point), k=1)
        return int(self._routable[int(index[0])]), float(distance[0])
