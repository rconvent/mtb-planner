"""Reading and writing the on-disk routing artifact.

A directory of plain `.npy` files plus a manifest. Rejected alternatives:
pickle executes arbitrary code on load and breaks across versions; SQLite
charges per-row overhead for data that is only ever used whole-array;
Parquet needs pyarrow and decodes into fresh buffers anyway.

Arrays are read fully into memory rather than memory-mapped. Windows locks a
mapped file, which would make the artifact impossible to rebuild while a
server holds it — a papercut that costs more than the sharing saves for a
single-process deployment.

Coordinates are stored as `int32` scaled to 1e-7 degrees, matching OSM's own
encoding, and widened to `float64` on load. That halves the largest array on
disk and round-trips exactly, which `float32` degrees would not.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mtb_planner.config.logging import get_logger
from mtb_planner.core.errors import ArtifactNotFoundError
from mtb_planner.core.units import COORD_SCALE
from mtb_planner.elevation.profile import EdgeElevation
from mtb_planner.features.model import EdgeFeatures
from mtb_planner.graph.manifest import (
    GraphCounts,
    Manifest,
    SourceExtract,
    build_manifest,
    read_manifest,
)
from mtb_planner.graph.store import GraphStore

logger = get_logger(__name__)

#: Arrays written verbatim, with the dtype each is restored as.
_ARRAYS: dict[str, np.dtype[np.generic]] = {
    "node_osm_id": np.dtype(np.int64),
    "node_scc": np.dtype(np.int32),
    "edge_u": np.dtype(np.int32),
    "edge_v": np.dtype(np.int32),
    "edge_length_m": np.dtype(np.float64),
    "edge_way_id": np.dtype(np.int64),
    "edge_access": np.dtype(np.uint8),
    "edge_geom_offset": np.dtype(np.int64),
    "indptr": np.dtype(np.int32),
    "arc_head": np.dtype(np.int32),
    "arc_edge": np.dtype(np.int32),
    "arc_rev": np.dtype(np.int32),
    "arc_allowed": np.dtype(np.uint8),
    "arc_forward": np.dtype(np.uint8),
}

#: Per-edge features, stored under a prefix so they cannot collide with the
#: structural arrays above.
_FEATURE_ARRAYS: dict[str, np.dtype[np.generic]] = {
    "tech": np.dtype(np.float32),
    "tech_up": np.dtype(np.float32),
    "tech_conf": np.dtype(np.uint8),
    "singletrack": np.dtype(np.float32),
    "rollability": np.dtype(np.float32),
    "flow": np.dtype(np.float32),
    "traffic": np.dtype(np.float32),
    "surface_class": np.dtype(np.uint8),
    "way_class": np.dtype(np.uint8),
    "n_mtb_routes": np.dtype(np.uint8),
    "route_flags": np.dtype(np.uint8),
}

#: Feature columns are namespaced on disk so a future column cannot collide
#: with a graph array.
_FEATURE_PREFIX = "feat_"

#: Elevation columns, likewise namespaced. `grade_hist` is two-dimensional:
#: one row of band lengths per edge.
_ELEVATION_PREFIX = "elev_"
_ELEVATION_ARRAYS: dict[str, np.dtype[np.generic]] = {
    "ascent_m": np.dtype(np.float32),
    "descent_m": np.dtype(np.float32),
    "grade_max": np.dtype(np.float32),
    "grade_min": np.dtype(np.float32),
    "grade_hist": np.dtype(np.int16),
}

#: Coordinates, stored scaled and widened on load.
_SCALED_ARRAYS = ("node_lat_e7", "node_lon_e7", "geom_e7")


def _to_e7(degrees: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    """Encode WGS84 degrees as `int32` at 1e-7 resolution."""
    return np.rint(degrees * COORD_SCALE).astype(np.int32)


def graph_counts(store: GraphStore) -> GraphCounts:
    """Summarise a graph's shape, including how much of it is reachable."""
    in_giant = (store.node_scc[store.edge_u] == store.giant_scc) & (
        store.node_scc[store.edge_v] == store.giant_scc
    )
    return GraphCounts(
        nodes=store.n_nodes,
        edges=store.n_edges,
        arcs=store.n_arcs,
        allowed_arcs=store.n_allowed_arcs,
        geom_points=int(store.geom.shape[0]),
        giant_scc_nodes=store.giant_scc_size,
        giant_scc_edges=int(in_giant.sum()),
    )


def artifact_size_bytes(directory: Path) -> int:
    """Total size on disk of an artifact directory."""
    return sum(path.stat().st_size for path in directory.glob("*") if path.is_file())


def save_artifact(
    store: GraphStore,
    directory: Path,
    artifact: str,
    contracted: bool = True,
    sources: tuple[SourceExtract, ...] = (),
) -> Manifest:
    """Write a graph and its manifest to an artifact directory."""
    directory.mkdir(parents=True, exist_ok=True)

    columns: dict[str, npt.NDArray[np.generic]] = {
        "node_osm_id": store.node_osm_id,
        "node_scc": store.node_scc,
        "edge_u": store.edge_u,
        "edge_v": store.edge_v,
        "edge_length_m": store.edge_length_m,
        "edge_way_id": store.edge_way_id,
        "edge_access": store.edge_access,
        "edge_geom_offset": store.edge_geom_offset,
        "indptr": store.indptr,
        "arc_head": store.arc_head,
        "arc_edge": store.arc_edge,
        "arc_rev": store.arc_rev,
        "arc_allowed": store.arc_allowed,
        "arc_forward": store.arc_forward,
    }
    for name, dtype in _ARRAYS.items():
        np.save(directory / f"{name}.npy", columns[name].astype(dtype, copy=False))

    for name, dtype in _FEATURE_ARRAYS.items():
        values = getattr(store.features, name)
        np.save(directory / f"{_FEATURE_PREFIX}{name}.npy", values.astype(dtype, copy=False))

    np.save(directory / "node_lat_e7.npy", _to_e7(store.node_lat))
    np.save(directory / "node_lon_e7.npy", _to_e7(store.node_lon))
    np.save(directory / "geom_e7.npy", _to_e7(store.geom))

    for name, dtype in _FEATURE_ARRAYS.items():
        column = getattr(store.features, name)
        np.save(directory / f"{_FEATURE_PREFIX}{name}.npy", column.astype(dtype, copy=False))

    for name, dtype in _ELEVATION_ARRAYS.items():
        column = getattr(store.elevation, name)
        np.save(directory / f"{_ELEVATION_PREFIX}{name}.npy", column.astype(dtype, copy=False))

    box = store.bbox()
    manifest = build_manifest(
        artifact=artifact,
        bbox=(box.min_lon, box.min_lat, box.max_lon, box.max_lat),
        counts=graph_counts(store),
        giant_scc=store.giant_scc,
        contracted=contracted,
        sources=sources,
        tech_coverage=store.length_by_confidence(),
        total_ascent_m=store.elevation.total_ascent_m,
    )
    manifest.write(directory)

    logger.info(
        "Wrote artifact %s to %s (%.1f MB)",
        artifact,
        directory,
        artifact_size_bytes(directory) / 1e6,
    )
    return manifest


def load_artifact(directory: Path) -> tuple[GraphStore, Manifest]:
    """Load an artifact, refusing anything this build cannot interpret."""
    manifest = read_manifest(directory)
    manifest.require_compatible()

    started = time.perf_counter()
    expected = (
        *_ARRAYS,
        *_SCALED_ARRAYS,
        *(f"{_FEATURE_PREFIX}{name}" for name in _FEATURE_ARRAYS),
        *(f"{_ELEVATION_PREFIX}{name}" for name in _ELEVATION_ARRAYS),
    )
    missing = [name for name in expected if not (directory / f"{name}.npy").exists()]
    if missing:
        msg = f"Artifact at {directory} is incomplete; missing {', '.join(sorted(missing))}"
        raise ArtifactNotFoundError(msg)

    def read(name: str) -> Any:
        """Load one array, coerced to the dtype this build expects.

        `.npy` carries its own dtype, which mypy cannot see, so the return is
        untyped by necessity. Coercing against `_ARRAYS` makes the runtime
        contract explicit instead: an array stored as a different width cannot
        quietly change the arithmetic that reads it.
        """
        values = np.load(directory / f"{name}.npy", allow_pickle=False)
        dtype = _ARRAYS.get(name) or _FEATURE_ARRAYS.get(name.removeprefix(_FEATURE_PREFIX))
        return values if dtype is None else values.astype(dtype, copy=False)

    features = EdgeFeatures(**{name: read(f"{_FEATURE_PREFIX}{name}") for name in _FEATURE_ARRAYS})
    elevation = EdgeElevation(
        **{name: read(f"{_ELEVATION_PREFIX}{name}") for name in _ELEVATION_ARRAYS}
    )

    store = GraphStore(
        node_lat=read("node_lat_e7").astype(np.float64) / COORD_SCALE,
        node_lon=read("node_lon_e7").astype(np.float64) / COORD_SCALE,
        node_osm_id=read("node_osm_id"),
        node_scc=read("node_scc"),
        edge_u=read("edge_u"),
        edge_v=read("edge_v"),
        edge_length_m=read("edge_length_m"),
        edge_way_id=read("edge_way_id"),
        edge_access=read("edge_access"),
        edge_geom_offset=read("edge_geom_offset"),
        geom=read("geom_e7").astype(np.float64) / COORD_SCALE,
        features=features,
        elevation=elevation,
        indptr=read("indptr"),
        arc_head=read("arc_head"),
        arc_edge=read("arc_edge"),
        arc_rev=read("arc_rev"),
        arc_allowed=read("arc_allowed"),
        arc_forward=read("arc_forward"),
        giant_scc=manifest.giant_scc,
    )

    logger.info(
        "Loaded %s: %d nodes, %d edges in %.2fs",
        manifest.artifact,
        store.n_nodes,
        store.n_edges,
        time.perf_counter() - started,
    )
    return store, manifest
