"""The graph registry.

A failed load must be recorded rather than raised, so the server can stay up
and explain itself through the health endpoint.
"""

from __future__ import annotations

import pytest

from mtb_planner import ARTIFACT_SCHEMA_VERSION, COST_MODEL_VERSION
from mtb_planner.core.errors import GraphNotLoadedError
from mtb_planner.graph.manifest import build_manifest
from mtb_planner.graph.serialize import graph_counts
from mtb_planner.graph.spatial_index import SpatialIndex
from mtb_planner.graph.store import GraphStore
from mtb_planner.services.graph_registry import GraphRegistry, LoadedGraph, get_registry

pytestmark = pytest.mark.unit


@pytest.fixture
def loaded(ardennes_graph: GraphStore, ardennes_index: SpatialIndex) -> LoadedGraph:
    """A `LoadedGraph` built from the fixture network."""
    box = ardennes_graph.bbox()
    manifest = build_manifest(
        artifact="ardennes-test",
        bbox=(box.min_lon, box.min_lat, box.max_lon, box.max_lat),
        counts=graph_counts(ardennes_graph),
        giant_scc=ardennes_graph.giant_scc,
        contracted=True,
    )
    return LoadedGraph.from_artifact(ardennes_graph, ardennes_index, manifest)


def test_a_fresh_registry_reports_not_loaded(registry: GraphRegistry):
    state = registry.state()

    assert registry.is_loaded is False
    assert state.loaded is False
    assert state.error is None


def test_require_raises_when_no_graph_is_loaded(registry: GraphRegistry):
    with pytest.raises(GraphNotLoadedError, match="not loaded"):
        registry.require()


def test_require_surfaces_the_recorded_failure_reason(registry: GraphRegistry):
    # Arrange
    registry.set_error("No routing artifact at data/graph/aoi-v1")

    # Act / Assert — the operator needs the reason, not a generic 503.
    with pytest.raises(GraphNotLoadedError, match="No routing artifact"):
        registry.require()


def test_setting_an_error_is_reported_in_state(registry: GraphRegistry):
    registry.set_error("boom")

    state = registry.state()

    assert state.loaded is False
    assert state.error == "boom"


def test_a_loaded_graph_is_returned_by_require(registry: GraphRegistry, loaded: LoadedGraph):
    registry.set_graph(loaded)

    assert registry.require() is loaded
    assert registry.is_loaded is True


def test_a_loaded_graph_can_actually_route(loaded: LoadedGraph):
    # The point of holding the store and index rather than metadata alone.
    assert loaded.store.n_edges > 0
    assert loaded.index.n_routable_nodes > 0


def test_state_reports_the_loaded_graphs_provenance(
    registry: GraphRegistry, loaded: LoadedGraph, ardennes_graph: GraphStore
):
    registry.set_graph(loaded)

    state = registry.state()

    assert state.loaded is True
    assert state.artifact == "ardennes-test"
    assert state.schema_version == ARTIFACT_SCHEMA_VERSION
    assert state.cost_model_version == COST_MODEL_VERSION
    assert state.nodes == ardennes_graph.n_nodes
    assert state.edges == ardennes_graph.n_edges
    assert state.loaded_at is not None
    assert state.loaded_at.endswith("Z")


def test_loading_a_graph_clears_a_previous_failure(registry: GraphRegistry, loaded: LoadedGraph):
    registry.set_error("transient problem")

    registry.set_graph(loaded)

    assert registry.state().error is None


def test_clear_drops_both_graph_and_error(registry: GraphRegistry, loaded: LoadedGraph):
    registry.set_graph(loaded)

    registry.clear()

    assert registry.is_loaded is False
    assert registry.state().error is None


@pytest.mark.parametrize(
    ("lat", "lon", "expected"),
    [
        (50.101, 5.101, True),  # centre of the fixture grid
        (48.000, 2.000, False),  # far outside
        (50.101, 8.000, False),  # east of coverage
    ],
)
def test_covers_checks_the_artifact_bounding_box(loaded: LoadedGraph, lat, lon, expected):
    assert loaded.covers(lat, lon) is expected


def test_get_registry_returns_the_same_instance():
    assert get_registry() is get_registry()
