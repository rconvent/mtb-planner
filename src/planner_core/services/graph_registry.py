"""Process-wide holder for the loaded routing graph.

The graph is large and loaded once at startup, so it lives in a module-level
singleton rather than being threaded through call sites. Access goes through
`get_registry()`, which is also the seam tests override.

A failed load is recorded rather than raised: the server stays up in a
degraded state so `/api/health` can explain what went wrong, instead of dying
before it can serve the diagnosis.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from planner_core.api.schemas.health import GraphState
from planner_core.core.errors import GraphNotLoadedError
from planner_core.core.geo import BBox
from planner_core.graph.manifest import Manifest
from planner_core.graph.spatial_index import SpatialIndex
from planner_core.graph.store import GraphStore


@dataclass(frozen=True)
class LoadedGraph:
    """A routing graph resident in memory, with its provenance.

    Holds the searchable graph and its spatial index, not just metadata: a
    "loaded" graph that cannot answer a query is not loaded.
    """

    artifact: str
    schema_version: int
    cost_model_version: int
    store: GraphStore
    index: SpatialIndex
    loaded_at: datetime

    @classmethod
    def from_artifact(
        cls, store: GraphStore, index: SpatialIndex, manifest: Manifest
    ) -> LoadedGraph:
        """Build from a deserialised artifact and its manifest."""
        return cls(
            artifact=manifest.artifact,
            schema_version=manifest.schema_version,
            cost_model_version=manifest.cost_model_version,
            store=store,
            index=index,
            loaded_at=datetime.now(UTC),
        )

    @property
    def n_nodes(self) -> int:
        """Number of graph nodes."""
        return self.store.n_nodes

    @property
    def n_edges(self) -> int:
        """Number of graph edges."""
        return self.store.n_edges

    @property
    def bbox(self) -> BBox:
        """Area the artifact covers."""
        return self.store.bbox()

    def covers(self, lat: float, lon: float) -> bool:
        """Whether a coordinate falls inside the artifact's bounding box.

        Checked before touching the graph so an out-of-area request fails
        immediately with a clear reason rather than as 'no route found'.
        """
        box = self.bbox
        return box.min_lat <= lat <= box.max_lat and box.min_lon <= lon <= box.max_lon


class GraphRegistry:
    """Holds at most one `LoadedGraph`, or the reason there is none."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._graph: LoadedGraph | None = None
        self._error: str | None = None

    def set_graph(self, graph: LoadedGraph) -> None:
        """Publish a loaded graph, clearing any previous failure."""
        with self._lock:
            self._graph = graph
            self._error = None

    def set_error(self, message: str) -> None:
        """Record why no graph is available."""
        with self._lock:
            self._graph = None
            self._error = message

    def clear(self) -> None:
        """Drop any graph and any recorded failure."""
        with self._lock:
            self._graph = None
            self._error = None

    @property
    def is_loaded(self) -> bool:
        """Whether a graph is resident."""
        return self._graph is not None

    def require(self) -> LoadedGraph:
        """Return the loaded graph, or raise if there is none.

        Every routing path calls this, so the 503 story is uniform.
        """
        graph = self._graph
        if graph is None:
            raise GraphNotLoadedError(self._error or "Routing graph is not loaded")
        return graph

    def state(self) -> GraphState:
        """Summarise current state for the health endpoint."""
        graph = self._graph
        if graph is None:
            return GraphState(loaded=False, error=self._error)
        return GraphState(
            loaded=True,
            artifact=graph.artifact,
            schema_version=graph.schema_version,
            cost_model_version=graph.cost_model_version,
            nodes=graph.n_nodes,
            edges=graph.n_edges,
            bbox=tuple(graph.bbox),
            loaded_at=graph.loaded_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )


_REGISTRY = GraphRegistry()


def get_registry() -> GraphRegistry:
    """Return the process-wide graph registry."""
    return _REGISTRY
