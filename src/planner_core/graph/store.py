"""The in-memory routing graph.

Flat NumPy arrays in compressed-sparse-row form, never a `networkx` graph. At
the scale this project targets — roughly 11 million directed arcs — NetworkX
costs 0.7–1 KB per edge in dictionary overhead, which is 8–12 GB before any
routing happens.

Arcs, not edges, are what searches traverse. Each undirected edge contributes
two arcs, and `arc_rev` links each to its twin. Storing the *arc* in a search's
predecessor array is what makes U-turn prohibition and turn costs expressible
at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from planner_core.core.geo import BBox, LatLon
from planner_core.elevation.profile import EdgeElevation
from planner_core.features.enums import TechConfidence
from planner_core.features.model import EdgeFeatures


@dataclass(frozen=True)
class GraphStore:
    """A routable graph held entirely in memory."""

    # --- nodes -----------------------------------------------------------
    node_lat: npt.NDArray[np.float64]
    node_lon: npt.NDArray[np.float64]
    node_osm_id: npt.NDArray[np.int64]
    node_scc: npt.NDArray[np.int32]

    # --- edges (undirected) ----------------------------------------------
    edge_u: npt.NDArray[np.int32]
    edge_v: npt.NDArray[np.int32]
    edge_length_m: npt.NDArray[np.float64]
    edge_way_id: npt.NDArray[np.int64]
    edge_access: npt.NDArray[np.uint8]
    edge_geom_offset: npt.NDArray[np.int64]
    geom: npt.NDArray[np.float64]

    #: What each edge is like to ride: technicality, surface, singletrack,
    #: signed-route membership. Positional against the edge arrays above.
    features: EdgeFeatures

    #: How much each edge climbs and how steeply. Positional against the edge
    #: arrays above.
    elevation: EdgeElevation

    # --- arcs (directed, CSR by source node) ------------------------------
    indptr: npt.NDArray[np.int32]
    arc_head: npt.NDArray[np.int32]
    arc_edge: npt.NDArray[np.int32]
    arc_rev: npt.NDArray[np.int32]
    arc_allowed: npt.NDArray[np.uint8]
    arc_forward: npt.NDArray[np.uint8]

    #: Label of the largest strongly connected component. Snapping outside it
    #: produces "no route found", which is why it is recorded rather than
    #: rediscovered per request.
    giant_scc: int

    @property
    def n_nodes(self) -> int:
        """Number of graph nodes."""
        return int(self.node_lat.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of undirected edges."""
        return int(self.edge_u.shape[0])

    @property
    def n_arcs(self) -> int:
        """Number of directed arcs, including those currently disallowed."""
        return int(self.arc_head.shape[0])

    @property
    def n_allowed_arcs(self) -> int:
        """Number of arcs a search may actually traverse."""
        return int(self.arc_allowed.sum())

    @property
    def giant_scc_size(self) -> int:
        """Number of nodes in the largest strongly connected component."""
        return int(np.count_nonzero(self.node_scc == self.giant_scc))

    def bbox(self) -> BBox:
        """Bounding box covering every node."""
        return BBox(
            min_lon=float(self.node_lon.min()),
            min_lat=float(self.node_lat.min()),
            max_lon=float(self.node_lon.max()),
            max_lat=float(self.node_lat.max()),
        )

    def node_point(self, node: int) -> LatLon:
        """Coordinate of a node."""
        return LatLon(lat=float(self.node_lat[node]), lon=float(self.node_lon[node]))

    def in_giant_component(self, node: int) -> bool:
        """Whether a node can reach, and be reached from, the main network."""
        return bool(self.node_scc[node] == self.giant_scc)

    def edge_geometry(self, edge: int, *, forward: bool = True) -> npt.NDArray[np.float64]:
        """Return an edge's `(N, 2)` lat/lon polyline, optionally reversed.

        The returned array is a view onto the shared geometry blob when
        `forward` is true; callers must not write to it.
        """
        start = int(self.edge_geom_offset[edge])
        stop = int(self.edge_geom_offset[edge + 1])
        points = self.geom[start:stop]
        return points if forward else points[::-1]

    def arc_geometry(self, arc: int) -> npt.NDArray[np.float64]:
        """Return an arc's polyline, oriented along the direction of travel."""
        edge = int(self.arc_edge[arc])
        return self.edge_geometry(edge, forward=bool(self.arc_forward[arc]))

    def arc_length_m(self, arc: int) -> float:
        """Length of the edge an arc traverses."""
        return float(self.edge_length_m[self.arc_edge[arc]])

    def arc_tail(self, arc: int) -> int:
        """Source node of an arc."""
        edge = int(self.arc_edge[arc])
        return int(self.edge_u[edge] if self.arc_forward[arc] else self.edge_v[edge])

    def arcs_of(self, node: int) -> range:
        """Arc indices leaving a node."""
        return range(int(self.indptr[node]), int(self.indptr[node + 1]))

    def total_length_m(self) -> float:
        """Summed length of every edge."""
        return float(self.edge_length_m.sum())

    def length_by_confidence(self) -> dict[str, float]:
        """Share of network length at each difficulty-confidence level.

        Measured in metres rather than edge counts, so a thousand short urban
        links cannot flatter the figure.
        """
        total = self.total_length_m()
        if total <= 0.0:
            return {level.name.lower(): 0.0 for level in TechConfidence}
        return {
            level.name.lower(): float(
                self.edge_length_m[self.features.tech_conf == int(level)].sum()
            )
            / total
            for level in TechConfidence
        }
