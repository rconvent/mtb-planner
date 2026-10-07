"""Artifact persistence and its compatibility gate.

Serialisation drift is the most likely silent breakage in this system: an
artifact that loads but decodes slightly differently produces wrong routes
with no error anywhere. The round-trip test therefore compares every array,
not a summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mtb_planner import COST_MODEL_VERSION
from planner_core import ARTIFACT_SCHEMA_VERSION
from planner_core.core.errors import ArtifactNotFoundError, IncompatibleArtifactError
from planner_core.core.geo import LatLon
from planner_core.graph.manifest import MANIFEST_FILENAME, read_manifest
from planner_core.graph.serialize import graph_counts, load_artifact, save_artifact
from planner_core.graph.spatial_index import SpatialIndex
from planner_core.graph.store import GraphStore
from planner_core.routing.point_to_point import plan_point_to_point
from tests.fixtures.networks import GRID_NW, GRID_SE

pytestmark = pytest.mark.unit

#: Arrays that must survive a round trip untouched.
_INTEGER_ARRAYS = (
    "node_osm_id",
    "node_scc",
    "edge_u",
    "edge_v",
    "edge_way_id",
    "edge_access",
    "edge_geom_offset",
    "indptr",
    "arc_head",
    "arc_edge",
    "arc_rev",
    "arc_allowed",
    "arc_forward",
)


@pytest.fixture
def artifact(ardennes_graph: GraphStore, tmp_path: Path) -> Path:
    """A written artifact directory."""
    directory = tmp_path / "ardennes-v1"
    save_artifact(ardennes_graph, directory, artifact="ardennes-v1")
    return directory


def test_saving_writes_a_manifest_and_arrays(artifact: Path):
    assert (artifact / MANIFEST_FILENAME).exists()
    assert len(list(artifact.glob("*.npy"))) >= len(_INTEGER_ARRAYS)


def test_the_manifest_is_valid_json(artifact: Path):
    payload = json.loads((artifact / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert payload["artifact"] == "ardennes-v1"
    assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert payload["cost_model_version"] == COST_MODEL_VERSION


def test_the_manifest_timestamp_is_utc_iso8601(artifact: Path):
    built_at = read_manifest(artifact).built_at

    assert built_at.endswith("Z")
    assert len(built_at) == len("2026-08-11T14:52:00Z")


@pytest.mark.parametrize("name", _INTEGER_ARRAYS)
def test_integer_arrays_round_trip_exactly(artifact: Path, ardennes_graph: GraphStore, name):
    restored, _ = load_artifact(artifact)

    np.testing.assert_array_equal(getattr(restored, name), getattr(ardennes_graph, name))


def test_lengths_round_trip_exactly(artifact: Path, ardennes_graph: GraphStore):
    restored, _ = load_artifact(artifact)

    np.testing.assert_array_equal(restored.edge_length_m, ardennes_graph.edge_length_m)


def test_coordinates_round_trip_within_their_stored_precision(
    artifact: Path, ardennes_graph: GraphStore
):
    # Stored as int32 at 1e-7 degrees, which is roughly a centimetre.
    restored, _ = load_artifact(artifact)

    np.testing.assert_allclose(restored.node_lat, ardennes_graph.node_lat, atol=1e-7)
    np.testing.assert_allclose(restored.node_lon, ardennes_graph.node_lon, atol=1e-7)
    np.testing.assert_allclose(restored.geom, ardennes_graph.geom, atol=1e-7)


def test_the_restored_graph_reports_the_same_shape(artifact: Path, ardennes_graph: GraphStore):
    restored, _ = load_artifact(artifact)

    assert restored.n_nodes == ardennes_graph.n_nodes
    assert restored.n_edges == ardennes_graph.n_edges
    assert restored.n_arcs == ardennes_graph.n_arcs
    assert restored.n_allowed_arcs == ardennes_graph.n_allowed_arcs
    assert restored.giant_scc_size == ardennes_graph.giant_scc_size


def test_the_restored_graph_routes_identically(artifact: Path, ardennes_graph: GraphStore, profile):
    # The end that matters: a rider must get the same ride from an artifact as
    # from a freshly built graph.
    restored, _ = load_artifact(artifact)

    plans = [
        plan_point_to_point(
            store,
            SpatialIndex(store),
            LatLon(*GRID_NW),
            LatLon(*GRID_SE),
            max_snap_distance_m=250.0,
            profile=profile,
        )
        for store in (ardennes_graph, restored)
    ]

    assert plans[0].route.arcs == plans[1].route.arcs
    assert plans[0].route.stats.distance_m == pytest.approx(plans[1].route.stats.distance_m)


def test_counts_record_how_much_of_the_network_is_reachable(ardennes_graph: GraphStore):
    counts = graph_counts(ardennes_graph)

    assert counts.edges == ardennes_graph.n_edges
    assert 0 < counts.giant_scc_edges <= counts.edges
    assert 0.0 < counts.giant_scc_edge_fraction <= 1.0


# --------------------------------------------------------------- the gate


def test_loading_a_missing_artifact_names_the_directory(tmp_path: Path):
    with pytest.raises(ArtifactNotFoundError, match="No routing artifact"):
        load_artifact(tmp_path / "absent")


def test_an_incomplete_artifact_is_refused(artifact: Path):
    # A half-written artifact must not load as if it were whole.
    (artifact / "arc_rev.npy").unlink()

    with pytest.raises(ArtifactNotFoundError, match="incomplete"):
        load_artifact(artifact)


def _rewrite_manifest_field(artifact: Path, field: str, value: int) -> None:
    path = artifact / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_a_newer_schema_version_refuses_to_load(artifact: Path):
    # This is the cheapest insurance in the project: a mismatch here would
    # otherwise produce plausible-looking wrong routes with no error at all.
    _rewrite_manifest_field(artifact, "schema_version", ARTIFACT_SCHEMA_VERSION + 1)

    with pytest.raises(IncompatibleArtifactError, match="schema"):
        load_artifact(artifact)


def test_a_different_cost_model_version_refuses_to_load(artifact: Path):
    _rewrite_manifest_field(artifact, "cost_model_version", COST_MODEL_VERSION + 1)

    with pytest.raises(IncompatibleArtifactError, match="cost model"):
        load_artifact(artifact)


def test_the_refusal_says_how_to_recover(artifact: Path):
    _rewrite_manifest_field(artifact, "schema_version", ARTIFACT_SCHEMA_VERSION + 1)

    with pytest.raises(IncompatibleArtifactError, match="mtb build"):
        load_artifact(artifact)


def test_a_corrupt_manifest_is_reported_clearly(artifact: Path):
    (artifact / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(IncompatibleArtifactError, match="not readable"):
        read_manifest(artifact)


def test_a_manifest_missing_required_fields_is_reported_clearly(artifact: Path):
    (artifact / MANIFEST_FILENAME).write_text('{"artifact": "x"}', encoding="utf-8")

    with pytest.raises(IncompatibleArtifactError, match="not readable"):
        read_manifest(artifact)


def test_overwriting_an_artifact_in_place_works(ardennes_graph: GraphStore, artifact: Path):
    # Rebuilding into the same directory is the normal case, so it must not
    # need the old one removed first.
    save_artifact(ardennes_graph, artifact, artifact="ardennes-v1")

    restored, manifest = load_artifact(artifact)
    assert manifest.artifact == "ardennes-v1"
    assert restored.n_edges == ardennes_graph.n_edges


# --------------------------------------------------------------- which artifact


def _build(directory: Path, name: str) -> Path:
    """A directory that looks built, as far as resolution is concerned."""
    artifact = directory / name
    artifact.mkdir(parents=True)
    (artifact / "manifest.json").write_text("{}", encoding="utf-8")
    return artifact


def test_a_built_artifact_is_used_as_asked(tmp_path):
    from planner_core.graph.manifest import resolve_artifact_dir

    wanted = _build(tmp_path, "lux-v1")

    assert resolve_artifact_dir(wanted) == wanted


def test_the_only_artifact_present_is_served_under_another_name(tmp_path):
    """Refusing to serve the only built graph is pedantry, not safety.

    The configured name defaults to a guess about what somebody will build. When
    the guess is wrong the server used to come up, serve the interface, list the
    profiles and answer every single plan with 503 — four separate sessions ended
    that way before this rule existed.
    """
    from planner_core.graph.manifest import resolve_artifact_dir

    built = _build(tmp_path, "lux-v1")

    assert resolve_artifact_dir(tmp_path / "aoi-v1") == built


def test_two_artifacts_are_never_guessed_between(tmp_path):
    """The half of the rule that matters more.

    Picking one of several is how somebody plans a route against last month's
    graph and never notices. With any ambiguity the configured path comes back
    untouched, so the caller fails and names them.
    """
    from planner_core.graph.manifest import resolve_artifact_dir

    _build(tmp_path, "lux-v1")
    _build(tmp_path, "aoi-v2")

    wanted = tmp_path / "aoi-v1"
    assert resolve_artifact_dir(wanted) == wanted


def test_nothing_built_resolves_to_what_was_asked_for(tmp_path):
    from planner_core.graph.manifest import resolve_artifact_dir

    wanted = tmp_path / "aoi-v1"

    assert resolve_artifact_dir(wanted) == wanted


def test_a_directory_without_a_manifest_is_not_an_artifact(tmp_path):
    # A half-written build, or somebody`s scratch folder.
    from planner_core.graph.manifest import resolve_artifact_dir

    (tmp_path / "not-an-artifact").mkdir()
    wanted = tmp_path / "aoi-v1"

    assert resolve_artifact_dir(wanted) == wanted
