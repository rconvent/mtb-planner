"""Contracting degree-2 chains.

After splitting, many nodes exist only because two OSM ways happen to meet
end to end. They are not junctions in any meaningful sense — a rider passes
straight through — and every one of them costs an edge, an arc pair, and a
settled node in every search that crosses it.

**What may be merged.** Two segments meeting at a node are merged only when
their entire retained tag set is identical, along with access, route
membership, and direction once both are oriented the same way. Tag equality
is stricter than it needs to be: the next phase derives features from tags,
so merging two ways whose tags differ would silently blend, say, a gravel
surface into a dirt one. Once features exist the key can loosen to feature
equality, which will contract considerably more.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mtb_planner.config.logging import get_logger
from mtb_planner.graph.segments import Segment
from mtb_planner.ingest.tables import NodeTable

logger = get_logger(__name__)

#: A node joins exactly two segments to be a candidate for removal.
_CHAIN_DEGREE = 2


def _endpoint_adjacency(
    segments: list[Segment], node_index: dict[int, int], n_nodes: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Build a CSR mapping from node to the segments touching it."""
    count = len(segments)
    endpoint_node = np.empty(2 * count, dtype=np.int64)
    endpoint_segment = np.empty(2 * count, dtype=np.int64)

    for index, segment in enumerate(segments):
        endpoint_node[index] = node_index[segment.u]
        endpoint_node[count + index] = node_index[segment.v]
        endpoint_segment[index] = index
        endpoint_segment[count + index] = index

    order = np.argsort(endpoint_node, kind="stable")
    degree = np.bincount(endpoint_node, minlength=n_nodes)
    indptr = np.concatenate([[0], np.cumsum(degree)]).astype(np.int64)
    return indptr, endpoint_segment[order], degree.astype(np.int64)


def _mergeable(first: Segment, second: Segment) -> bool:
    """Whether two segments, already oriented nose to tail, may become one."""
    return (
        first.direction == second.direction
        and first.access == second.access
        and first.routes == second.routes
        and first.tags == second.tags
    )


def _oriented_from(segment: Segment, node: int) -> Segment:
    """Return the segment oriented so that it starts at `node`."""
    return segment if segment.u == node else segment.reversed()


def _merge(chain: list[Segment]) -> Segment:
    """Join a nose-to-tail chain into a single segment."""
    if len(chain) == 1:
        return chain[0]
    parts = [chain[0].node_ids]
    parts.extend(segment.node_ids[1:] for segment in chain[1:])
    head = chain[0]
    return Segment(
        node_ids=np.concatenate(parts),
        # Provenance keeps the first way; the merge key guarantees every
        # contributing way carried identical tags.
        way_id=head.way_id,
        tags=head.tags,
        access=head.access,
        direction=head.direction,
        routes=head.routes,
    )


def contract(segments: list[Segment], nodes: NodeTable) -> list[Segment]:
    """Merge chains of segments that pass straight through a node."""
    if not segments:
        return []

    endpoints = sorted({segment.u for segment in segments} | {segment.v for segment in segments})
    node_index = {node_id: index for index, node_id in enumerate(endpoints)}
    n_nodes = len(endpoints)

    indptr, adjacency, degree = _endpoint_adjacency(segments, node_index, n_nodes)
    barrier_at = nodes.barrier[nodes.index_of(np.asarray(endpoints, dtype=np.int64))]

    def segments_at(node: int) -> list[int]:
        return [int(s) for s in adjacency[indptr[node] : indptr[node + 1]]]

    def passes_through(node: int) -> bool:
        """Whether a rider goes straight through, so the node can vanish."""
        if degree[node] != _CHAIN_DEGREE or barrier_at[node]:
            return False
        first, second = segments_at(node)
        if first == second:
            # A segment whose two ends meet here; removing the node would
            # leave an edge with nothing to attach to.
            return False
        node_id = endpoints[node]
        return _mergeable(
            _oriented_from(segments[first], node_id).reversed(),
            _oriented_from(segments[second], node_id),
        )

    through = np.fromiter(
        (passes_through(node) for node in range(n_nodes)), dtype=np.bool_, count=n_nodes
    )

    used = np.zeros(len(segments), dtype=np.bool_)
    merged: list[Segment] = []

    def follow(start_node: int, first_segment: int) -> Segment:
        """Walk a chain from a real junction until it reaches another one.

        A chain of one is returned untouched. Walking orients each segment
        away from the node it was reached from, which would otherwise flip
        lone segments according to which junction happened to be visited
        first — harmless, but it makes edges harder to compare against the
        OSM ways they came from.
        """
        chain: list[Segment] = []
        node_id = endpoints[start_node]
        segment_index = first_segment
        while True:
            used[segment_index] = True
            oriented = _oriented_from(segments[segment_index], node_id)
            chain.append(oriented)
            node_id = oriented.v
            far = node_index[node_id]
            if not through[far]:
                break
            candidates = [s for s in segments_at(far) if not used[s]]
            if not candidates:
                break
            segment_index = candidates[0]

        return segments[first_segment] if len(chain) == 1 else _merge(chain)

    for node in range(n_nodes):
        if through[node]:
            continue
        for segment_index in segments_at(node):
            if not used[segment_index]:
                merged.append(follow(node, segment_index))

    # Rings made entirely of pass-through nodes have no junction to start
    # from, so they are broken open at an arbitrary point.
    for segment_index in range(len(segments)):
        if not used[segment_index]:
            merged.append(follow(node_index[segments[segment_index].u], segment_index))

    removed = 100.0 * (1 - len(merged) / len(segments))
    logger.info(
        "Contracted %d segments into %d edges (%.1f%% removed)",
        len(segments),
        len(merged),
        removed,
    )
    return merged
