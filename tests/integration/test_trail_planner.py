"""The trail planner end to end: build, serve, plan.

The engine is shared and tested under the bike everywhere else. What needs
pinning here is that the sport actually changes things — the same extract
builds a different graph for a runner, the API speaks in a runner's terms, and
a graph built for one sport is refused by the other.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from mtb_planner.sport import MTB
from planner_core import ARTIFACT_SCHEMA_VERSION
from planner_core.api.app import create_app
from planner_core.api.deps import route_cache
from planner_core.core.errors import IncompatibleArtifactError
from planner_core.graph.builder import build_graph
from planner_core.graph.serialize import load_artifact, save_artifact
from planner_core.graph.spatial_index import SpatialIndex
from planner_core.graph.store import GraphStore
from planner_core.ingest.pbf_reader import read_extracts
from planner_core.routing.kernels import warm_up
from planner_core.services.graph_registry import LoadedGraph, get_registry
from planner_core.sport import Access, activate
from tests.fixtures.networks import FOOT_NE, FOOT_SW, LATTICE_CENTRE, footpaths
from trail import COST_MODEL_VERSION
from trail.cli import app as trail_cli
from trail.sport import TRAIL

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def footpaths_pbf(fixture_dir: Path) -> Path:
    return footpaths(fixture_dir)


@pytest.fixture(scope="session")
def trail_graph(footpaths_pbf: Path) -> GraphStore:
    return build_graph(read_extracts([footpaths_pbf], sport=TRAIL), sport=TRAIL)


@pytest.fixture(scope="session")
def bike_graph(footpaths_pbf: Path) -> GraphStore:
    return build_graph(read_extracts([footpaths_pbf], sport=MTB), sport=MTB)


@pytest.fixture(scope="session")
def trail_lattice(lattice_pbf: Path) -> GraphStore:
    return build_graph(read_extracts([lattice_pbf], sport=TRAIL), sport=TRAIL)


def edges_of(store: GraphStore, way_id: int) -> np.ndarray:
    return np.flatnonzero(store.edge_way_id == way_id)


def allowed_arcs_of(store: GraphStore, way_id: int) -> int:
    return int(store.arc_allowed[np.isin(store.arc_edge, edges_of(store, way_id))].sum())


def loaded(store: GraphStore, name: str) -> LoadedGraph:
    return LoadedGraph(
        artifact=name,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        cost_model_version=COST_MODEL_VERSION,
        store=store,
        index=SpatialIndex(store),
        loaded_at=datetime.now(UTC),
    )


@pytest.fixture
def trail_client(trail_graph: GraphStore) -> Iterator[TestClient]:
    warm_up()
    route_cache().clear()
    with TestClient(create_app(TRAIL)) as client:
        get_registry().set_graph(loaded(trail_graph, "footpaths-trail"))
        yield client
    route_cache().clear()


@pytest.fixture
def trail_lattice_client(trail_lattice: GraphStore) -> Iterator[TestClient]:
    warm_up()
    route_cache().clear()
    with TestClient(create_app(TRAIL)) as client:
        get_registry().set_graph(loaded(trail_lattice, "lattice-trail"))
        yield client
    route_cache().clear()


# --------------------------------------------------------------- the graph


def test_a_footpath_closed_to_bikes_is_a_runners(trail_graph: GraphStore, bike_graph: GraphStore):
    assert edges_of(trail_graph, 510).size > 0
    assert edges_of(bike_graph, 510).size == 0


def test_a_cycleway_closed_to_walkers_is_a_riders(trail_graph: GraphStore, bike_graph: GraphStore):
    assert edges_of(trail_graph, 511).size == 0
    assert edges_of(bike_graph, 511).size > 0


def test_a_one_way_descent_binds_only_the_bike(trail_graph: GraphStore, bike_graph: GraphStore):
    edges = edges_of(trail_graph, 512).size

    assert allowed_arcs_of(trail_graph, 512) == 2 * edges
    assert allowed_arcs_of(bike_graph, 512) == edges_of(bike_graph, 512).size


def test_steps_are_run_not_pushed(trail_graph: GraphStore, bike_graph: GraphStore):
    assert set(trail_graph.edge_access[edges_of(trail_graph, 513)]) == {Access.YES}
    assert set(bike_graph.edge_access[edges_of(bike_graph, 513)]) == {Access.WALK}


def test_a_scramble_is_walked(trail_graph: GraphStore):
    assert set(trail_graph.edge_access[edges_of(trail_graph, 514)]) == {Access.WALK}


def test_hiking_routes_are_signed_for_a_runner(trail_graph: GraphStore, bike_graph: GraphStore):
    on_route = edges_of(trail_graph, 515)

    assert trail_graph.features.signed[on_route].all()
    assert not bike_graph.features.signed[edges_of(bike_graph, 500)].any()


def test_sac_scale_is_surveyed_difficulty_on_foot(trail_graph: GraphStore):
    edges = edges_of(trail_graph, 515)

    assert np.allclose(trail_graph.features.tech[edges], 2.0)


# --------------------------------------------------------------- artifacts


def test_an_artifact_records_its_sport(trail_graph: GraphStore, tmp_path: Path):
    activate(TRAIL)
    manifest = save_artifact(trail_graph, tmp_path / "foot", artifact="foot")

    assert manifest.sport == "trail"
    assert manifest.cost_model_version == COST_MODEL_VERSION
    _, reread = load_artifact(tmp_path / "foot")
    assert reread.sport == "trail"


def test_a_runners_graph_is_refused_by_the_bike(trail_graph: GraphStore, tmp_path: Path):
    activate(TRAIL)
    save_artifact(trail_graph, tmp_path / "foot", artifact="foot")

    activate(MTB)
    with pytest.raises(IncompatibleArtifactError, match="built for trail"):
        load_artifact(tmp_path / "foot")


# --------------------------------------------------------------- the API


def test_the_runners_profiles_are_offered(trail_client: TestClient):
    data = trail_client.get("/api/profiles").json()["data"]

    assert {p["id"] for p in data["profiles"]} == {"easy", "trail", "technical", "vertical"}
    assert data["default_profile"] == "trail"


def test_effort_is_a_pace(trail_client: TestClient):
    sliders = trail_client.get("/api/profiles").json()["data"]["sliders"]
    effort = next(s for s in sliders if s["id"] == "effort")

    assert "min/km" in effort["label"]
    assert effort["min"] < effort["max"] <= 12


def test_the_page_speaks_a_runners_language(trail_client: TestClient):
    page = trail_client.get("/").text

    assert 'data-grade-prefix="T"' in page
    assert "Running" in page
    assert "mtb:scale" not in page


def test_a_route_between_two_points_is_planned(trail_client: TestClient):
    response = trail_client.post(
        "/api/routes/point-to-point",
        json={
            "start": {"lat": FOOT_SW[0], "lon": FOOT_SW[1]},
            "end": {"lat": FOOT_NE[0], "lon": FOOT_NE[1]},
            "effort": 5.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["profile"] == "trail"
    route = body["data"]["routes"][0]
    assert route["stats"]["distance_m"] > 0

    gpx = trail_client.get(route["gpx_url"])
    assert gpx.status_code == 200
    assert "trail-" in gpx.headers["content-disposition"]
    assert "trail-planner" in gpx.text


def test_a_bike_effort_is_not_a_pace(trail_client: TestClient):
    response = trail_client.post(
        "/api/routes/point-to-point",
        json={
            "start": {"lat": FOOT_SW[0], "lon": FOOT_SW[1]},
            "end": {"lat": FOOT_NE[0], "lon": FOOT_NE[1]},
            "effort": 250,
        },
    )

    assert response.status_code == 422


def test_a_faster_pace_is_a_quicker_route(trail_client: TestClient):
    def minutes(pace: float) -> float:
        body = {
            "start": {"lat": FOOT_SW[0], "lon": FOOT_SW[1]},
            "end": {"lat": FOOT_NE[0], "lon": FOOT_NE[1]},
            "effort": pace,
        }
        data = trail_client.post("/api/routes/point-to-point", json=body).json()["data"]
        return data["routes"][0]["stats"]["moving_time_s"]

    assert minutes(4.5) < minutes(8.0)


def test_loops_are_generated_for_a_runner(trail_lattice_client: TestClient):
    response = trail_lattice_client.post(
        "/api/routes/loop",
        json={
            "start": {"lat": LATTICE_CENTRE[0], "lon": LATTICE_CENTRE[1]},
            "budget_minutes": 30,
            "profile": "easy",
        },
    )

    assert response.status_code == 200
    loops = response.json()["data"]["loops"]
    assert loops
    assert all(loop["stats"]["moving_time_s"] > 0 for loop in loops)


# --------------------------------------------------------------- the CLI


def test_the_trail_command_names_itself():
    result = CliRunner().invoke(trail_cli, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.startswith("trail ")


def test_inspect_reports_a_runners_artifact(trail_graph: GraphStore, tmp_path: Path):
    activate(TRAIL)
    save_artifact(trail_graph, tmp_path / "foot", artifact="foot")

    result = CliRunner().invoke(trail_cli, ["inspect", str(tmp_path / "foot")])

    assert result.exit_code == 0, result.stdout
    assert "trail" in result.stdout
