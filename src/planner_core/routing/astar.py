"""A* over the CSR graph.

The predecessor array stores the *arc* a node was reached by, not the node.
That is what makes U-turn prohibition expressible: without it there is no way
to know which way the rider was facing on arrival.

This makes the search a heuristic in the strict sense — a node's optimal label
can differ per incoming arc, and being exact would mean searching the line
graph at twice the state. Every production router makes this trade; the
quality difference is negligible and the saving is a factor of two.

The heuristic is straight-line distance. While cost is measured in metres and
no edge is shorter than its own geometry, that is admissible, so the result is
optimal for the cost supplied. Once cost becomes time-based the heuristic must
be divided by the maximum plausible speed to stay admissible.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from planner_core.config.logging import get_logger
from planner_core.core.units import EARTH_RADIUS_M

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """The outcome of one search."""

    arcs: tuple[int, ...]
    cost: float
    settled: int
    reached: bool


def _haversine_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two coordinates in degrees."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def _reconstruct(
    pred_arc: npt.NDArray[np.int32],
    pred_node: npt.NDArray[np.int32],
    source: int,
    target: int,
) -> tuple[int, ...]:
    """Walk the predecessor arrays back from `target` to `source`."""
    arcs: list[int] = []
    node = target
    while node != source:
        arc = int(pred_arc[node])
        if arc < 0:
            msg = "Predecessor chain broke before reaching the source"
            raise RuntimeError(msg)
        arcs.append(arc)
        node = int(pred_node[node])
    arcs.reverse()
    return tuple(arcs)


def astar(
    *,
    indptr: npt.NDArray[np.int32],
    arc_head: npt.NDArray[np.int32],
    arc_rev: npt.NDArray[np.int32],
    arc_allowed: npt.NDArray[np.uint8],
    node_lat: npt.NDArray[np.float64],
    node_lon: npt.NDArray[np.float64],
    arc_cost: npt.NDArray[np.float64],
    source: int,
    target: int,
    forbid_u_turns: bool = True,
    heuristic_scale: float = 1.0,
    max_settled: int | None = None,
) -> SearchResult:
    """Find a least-cost arc sequence from `source` to `target`.

    `heuristic_scale` converts straight-line metres into the units of
    `arc_cost`. With cost in seconds that is the fastest second-per-metre
    anywhere in the graph, which keeps the heuristic a lower bound and so
    keeps the result optimal. `max_settled` bounds the work done, returning
    an unreached result rather than running forever on a disconnected request.
    """
    n_nodes = node_lat.shape[0]
    if source == target:
        return SearchResult(arcs=(), cost=0.0, settled=0, reached=True)

    target_lat = float(node_lat[target])
    target_lon = float(node_lon[target])

    def h(node: int) -> float:
        return (
            _haversine_deg(float(node_lat[node]), float(node_lon[node]), target_lat, target_lon)
            * heuristic_scale
        )

    dist = np.full(n_nodes, np.inf, dtype=np.float64)
    pred_arc = np.full(n_nodes, -1, dtype=np.int32)
    pred_node = np.full(n_nodes, -1, dtype=np.int32)
    dist[source] = 0.0

    # (estimated total, cost so far, node, arc arrived on). Tuple ordering
    # breaks ties on cost then node index, which keeps results reproducible.
    heap: list[tuple[float, float, int, int]] = [(h(source), 0.0, source, -1)]
    settled = 0

    while heap:
        _, cost, node, arrived_on = heapq.heappop(heap)
        if cost > dist[node]:
            continue
        if node == target:
            return SearchResult(
                arcs=_reconstruct(pred_arc, pred_node, source, target),
                cost=float(cost),
                settled=settled,
                reached=True,
            )

        settled += 1
        if max_settled is not None and settled > max_settled:
            logger.warning("A* gave up after settling %d nodes", settled)
            break

        for arc in range(int(indptr[node]), int(indptr[node + 1])):
            if not arc_allowed[arc]:
                continue
            if forbid_u_turns and arrived_on >= 0 and arc == arc_rev[arrived_on]:
                continue
            head = int(arc_head[arc])
            candidate = cost + float(arc_cost[arc])
            if candidate < dist[head]:
                dist[head] = candidate
                pred_arc[head] = arc
                pred_node[head] = node
                heapq.heappush(heap, (candidate + h(head), candidate, head, arc))

    return SearchResult(arcs=(), cost=math.inf, settled=settled, reached=False)
