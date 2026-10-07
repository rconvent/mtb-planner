"""Turns ingested ways into a routable CSR graph.

Four stages, in order: split ways at junctions, contract the chains that pass
straight through a node, assemble flat arrays, then derive directed arcs and
connectivity.

Geometry and length are computed once, vectorised across every segment at the
same time. Per-segment Python loops over a country's worth of edges are what
make a build take an hour instead of minutes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from planner_core.config.logging import get_logger
from planner_core.core.arrays import ragged_indices
from planner_core.core.errors import PbfParseError
from planner_core.core.geo import haversine_m_array
from planner_core.core.units import COORD_SCALE
from planner_core.elevation.attach import sample_edges
from planner_core.elevation.dem_sampler import DemSampler
from planner_core.elevation.profile import EdgeElevation
from planner_core.features.model import compute_features
from planner_core.features.propagate import propagate_difficulty
from planner_core.graph.components import strongly_connected
from planner_core.graph.segments import Segment, split_ways
from planner_core.graph.simplify import contract
from planner_core.graph.store import GraphStore
from planner_core.ingest.pbf_reader import IngestResult
from planner_core.ingest.tables import NodeTable
from planner_core.sport import Direction, Sport

logger = get_logger(__name__)


def _build_arcs(
    edge_u: npt.NDArray[np.int32],
    edge_v: npt.NDArray[np.int32],
    edge_direction: npt.NDArray[np.uint8],
    n_nodes: int,
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
]:
    """Assemble directed arcs in CSR order, with reverse-twin links.

    Arcs are laid out as all forward arcs then all backward arcs, so edge `e`
    has twins at `e` and `e + n_edges` before sorting. After sorting into CSR
    order the twin relationship is re-expressed through `arc_rev`.
    """
    n_edges = edge_u.shape[0]
    edge_index = np.arange(n_edges, dtype=np.int32)

    src = np.concatenate([edge_u, edge_v])
    dst = np.concatenate([edge_v, edge_u])
    arc_edge_unsorted = np.concatenate([edge_index, edge_index])
    forward = np.concatenate([np.ones(n_edges, dtype=np.uint8), np.zeros(n_edges, dtype=np.uint8)])

    allowed_forward = (edge_direction == Direction.BOTH) | (edge_direction == Direction.FORWARD)
    allowed_backward = (edge_direction == Direction.BOTH) | (edge_direction == Direction.BACKWARD)
    allowed = np.concatenate([allowed_forward, allowed_backward]).astype(np.uint8)

    # Sort by source node, breaking ties on destination so the layout — and
    # therefore every search that iterates it — is deterministic.
    order = np.lexsort((dst, src))
    src_sorted = src[order]

    position = np.empty(order.shape[0], dtype=np.int32)
    position[order] = np.arange(order.shape[0], dtype=np.int32)
    twin_unsorted = (np.arange(2 * n_edges, dtype=np.int32) + n_edges) % (2 * n_edges)

    counts = np.bincount(src_sorted, minlength=n_nodes)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)

    return (
        indptr,
        dst[order].astype(np.int32),
        arc_edge_unsorted[order].astype(np.int32),
        position[twin_unsorted[order]].astype(np.int32),
        allowed[order],
        forward[order],
    )


def _segment_geometry(
    segments: list[Segment], nodes: NodeTable
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Resolve every segment's polyline and length in one vectorised pass.

    Returns the concatenated `(P, 2)` lat/lon blob, the per-segment offsets
    into it, and the length of each segment in metres.
    """
    counts = np.fromiter(
        (segment.node_ids.shape[0] for segment in segments),
        dtype=np.int64,
        count=len(segments),
    )
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    positions = nodes.index_of(np.concatenate([segment.node_ids for segment in segments]))
    lat = nodes.lat_e7[positions] / COORD_SCALE
    lon = nodes.lon_e7[positions] / COORD_SCALE

    # Distance between every consecutive pair, then blank the pairs that
    # straddle two segments so each segment sums only its own geometry.
    step = haversine_m_array(lat[:-1], lon[:-1], lat[1:], lon[1:])
    step[offsets[1:-1] - 1] = 0.0
    lengths = np.add.reduceat(step, offsets[:-1])

    return np.column_stack((lat, lon)), offsets, lengths


def build_graph(
    ingest: IngestResult,
    contract_chains: bool = True,
    sampler: DemSampler | None = None,
    sport: Sport | None = None,
) -> GraphStore:
    """Build a routable graph from ingested ways and nodes.

    Without a `sampler` the graph is built flat. That is useful for tests and
    for proving the topology, but every distance-based figure it produces is
    a lower bound on the effort a ride actually costs.
    """
    segments = split_ways(ingest.nodes, ingest.ways, ingest.way_to_routes)
    if contract_chains:
        segments = contract(segments, ingest.nodes)
    if not segments:
        msg = "No segments survived the graph build"
        raise PbfParseError(msg)

    geometry, geom_offsets, lengths = _segment_geometry(segments, ingest.nodes)

    count = len(segments)
    u_ids = np.fromiter((s.u for s in segments), dtype=np.int64, count=count)
    v_ids = np.fromiter((s.v for s in segments), dtype=np.int64, count=count)

    # A self-loop carries nobody anywhere, and a zero-length edge makes the
    # search's cost ordering ambiguous.
    keep = np.flatnonzero((lengths > 0.0) & (u_ids != v_ids))
    if keep.shape[0] == 0:
        msg = "No edges survived the graph build"
        raise PbfParseError(msg)
    dropped = count - keep.shape[0]

    kept_u, kept_v = u_ids[keep], v_ids[keep]
    endpoint_ids = np.unique(np.concatenate([kept_u, kept_v]))
    edge_u = np.searchsorted(endpoint_ids, kept_u).astype(np.int32)
    edge_v = np.searchsorted(endpoint_ids, kept_v).astype(np.int32)
    edge_direction = np.fromiter((s.direction for s in segments), dtype=np.uint8, count=count)[keep]

    # Re-lay the geometry blob so the kept edges are contiguous.
    kept_counts = np.diff(geom_offsets)[keep]
    kept_offsets = np.concatenate([[0], np.cumsum(kept_counts)]).astype(np.int64)
    gathered = geometry[ragged_indices(geom_offsets[keep], kept_counts)]

    # Features are derived on the surviving edges only, and propagation runs
    # over the same set, so a group's surveyed fraction is measured against
    # what is actually in the graph.
    kept_segments = [segments[int(i)] for i in keep]
    kept_lengths = lengths[keep]
    features = propagate_difficulty(
        compute_features(kept_segments, ingest.route_relations, sport),
        kept_segments,
        kept_lengths,
    )

    if sampler is None:
        elevation = EdgeElevation.empty(keep.shape[0])
    else:
        structure = np.fromiter(
            (s.tags.is_bridge or s.tags.is_tunnel for s in kept_segments),
            dtype=np.bool_,
            count=len(kept_segments),
        )
        elevation = sample_edges(gathered, kept_offsets, structure, sampler)

    n_nodes = endpoint_ids.shape[0]
    source_positions = ingest.nodes.index_of(endpoint_ids)

    indptr, arc_head, arc_edge, arc_rev, arc_allowed, arc_forward = _build_arcs(
        edge_u, edge_v, edge_direction, n_nodes
    )
    node_scc, giant = strongly_connected(indptr, arc_head, arc_allowed, n_nodes)

    logger.info(
        "Graph: %d nodes, %d edges, %d/%d traversable arcs (%d degenerate segments dropped)",
        n_nodes,
        keep.shape[0],
        int(arc_allowed.sum()),
        arc_allowed.shape[0],
        dropped,
    )

    return GraphStore(
        node_lat=ingest.nodes.lat_e7[source_positions] / COORD_SCALE,
        node_lon=ingest.nodes.lon_e7[source_positions] / COORD_SCALE,
        node_osm_id=endpoint_ids,
        node_scc=node_scc,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_length_m=kept_lengths,
        edge_way_id=np.fromiter((s.way_id for s in segments), dtype=np.int64, count=count)[keep],
        edge_access=np.fromiter((s.access for s in segments), dtype=np.uint8, count=count)[keep],
        edge_geom_offset=kept_offsets,
        geom=gathered,
        elevation=elevation,
        features=features,
        indptr=indptr,
        arc_head=arc_head,
        arc_edge=arc_edge,
        arc_rev=arc_rev,
        arc_allowed=arc_allowed,
        arc_forward=arc_forward,
        giant_scc=giant,
    )
