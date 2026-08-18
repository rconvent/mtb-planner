"""Shared fixtures.

The graph registry is a process-wide singleton, so it must be reset between
tests or state leaks across them in whatever order pytest happens to run.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mtb_planner import ARTIFACT_SCHEMA_VERSION, COST_MODEL_VERSION
from mtb_planner.api.app import create_app
from mtb_planner.config.settings import get_settings
from mtb_planner.cost.profiles import RideProfile, get_profile
from mtb_planner.graph.builder import build_graph
from mtb_planner.graph.spatial_index import SpatialIndex
from mtb_planner.graph.store import GraphStore
from mtb_planner.ingest.pbf_reader import IngestResult, read_extracts
from mtb_planner.routing.kernels import warm_up
from mtb_planner.services.graph_registry import GraphRegistry, LoadedGraph, get_registry
from tests.fixtures.networks import ardennes_like, lattice


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Clear the graph registry before and after every test."""
    get_registry().clear()
    yield
    get_registry().clear()


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Drop the cached settings so env overrides take effect per test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _isolated_data_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the settings at an empty data directory.

    The server falls back to the only artifact present when the configured one
    is not built, which is right for a person with one graph on their laptop and
    wrong for a test: "no graph is loaded" would otherwise mean "no graph is
    loaded unless the developer happens to have built one", and three tests
    passed on CI while failing here. Tests that want a graph publish one into
    the registry themselves.
    """
    monkeypatch.setenv("MTB_DATA_DIR", str(tmp_path_factory.mktemp("empty-data")))
    get_settings.cache_clear()


@pytest.fixture
def registry() -> GraphRegistry:
    """A fresh, isolated registry — not the process singleton."""
    return GraphRegistry()


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Scratch directory holding generated OSM fixtures."""
    return tmp_path_factory.mktemp("osm")


@pytest.fixture(scope="session")
def ardennes_pbf(fixture_dir: Path) -> Path:
    """The workhorse fixture network, written once per test session."""
    return ardennes_like(fixture_dir)


@pytest.fixture(scope="session")
def ardennes_ingest(ardennes_pbf: Path) -> IngestResult:
    """Ingested ways, nodes, and relations from the fixture network."""
    return read_extracts([ardennes_pbf])


@pytest.fixture(scope="session")
def ardennes_graph(ardennes_ingest: IngestResult) -> GraphStore:
    """The fixture network built into a routable graph, as production builds it."""
    return build_graph(ardennes_ingest)


@pytest.fixture(scope="session")
def ardennes_graph_uncontracted(ardennes_ingest: IngestResult) -> GraphStore:
    """The same network with chain contraction disabled.

    Splitting semantics are far easier to assert against when every junction
    is still a node.
    """
    return build_graph(ardennes_ingest, contract_chains=False)


@pytest.fixture(scope="session")
def ardennes_index(ardennes_graph: GraphStore) -> SpatialIndex:
    """Spatial index over the fixture graph."""
    return SpatialIndex(ardennes_graph)


@pytest.fixture(scope="session")
def profile() -> RideProfile:
    """The shipped `flow` profile, used wherever a route needs one."""
    return get_profile("flow")


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A test client with the lifespan handler actually run.

    Entering the context manager is what triggers startup; without it the
    graph load never happens and health reports a state the server would
    never really be in.
    """
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def lattice_pbf(fixture_dir: Path) -> Path:
    """The larger lattice network, written once per session."""
    return lattice(fixture_dir)


@pytest.fixture(scope="session")
def lattice_graph(lattice_pbf: Path) -> GraphStore:
    """A network big enough to hold real loops.

    `ardennes_graph` is 1.5 km of trail, below the smallest budget the loop API
    accepts, so anything testing loop generation over HTTP needs this instead.
    """
    return build_graph(read_extracts([lattice_pbf]))


@pytest.fixture(scope="session")
def lattice_index(lattice_graph: GraphStore) -> SpatialIndex:
    """Spatial index over the lattice network."""
    return SpatialIndex(lattice_graph)


@pytest.fixture
def loaded_lattice(lattice_graph: GraphStore, lattice_index: SpatialIndex) -> LoadedGraph:
    """The lattice network dressed as a graph loaded from an artifact."""
    return LoadedGraph(
        artifact="lattice-fixture",
        schema_version=ARTIFACT_SCHEMA_VERSION,
        cost_model_version=COST_MODEL_VERSION,
        store=lattice_graph,
        index=lattice_index,
        loaded_at=datetime.now(UTC),
    )


@pytest.fixture
def lattice_client(loaded_lattice: LoadedGraph) -> Iterator[TestClient]:
    """A test client that can generate real loops."""
    warm_up()
    with TestClient(create_app()) as test_client:
        get_registry().set_graph(loaded_lattice)
        yield test_client


@pytest.fixture
def loaded_graph(ardennes_graph: GraphStore, ardennes_index: SpatialIndex) -> LoadedGraph:
    """The fixture network dressed as a graph loaded from an artifact.

    Built directly rather than round-tripped through disk: serialisation has
    its own tests, and the endpoints only care that a routable graph is
    resident.
    """
    return LoadedGraph(
        artifact="ardennes-fixture",
        schema_version=ARTIFACT_SCHEMA_VERSION,
        cost_model_version=COST_MODEL_VERSION,
        store=ardennes_graph,
        index=ardennes_index,
        loaded_at=datetime.now(UTC),
    )


@pytest.fixture
def routing_client(loaded_graph: LoadedGraph) -> Iterator[TestClient]:
    """A test client that can actually plan a route.

    The lifespan handler finds no artifact on disk and records that failure,
    so the graph is published afterwards — which is also what proves the
    endpoints read the registry per request rather than at startup.
    """
    warm_up()
    with TestClient(create_app()) as test_client:
        get_registry().set_graph(loaded_graph)
        yield test_client
