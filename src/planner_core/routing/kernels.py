"""Compiled search kernels.

The pure-Python A* is correct and readable, and on random long-range pairs
across Luxembourg it settles a median of 52,000 nodes at a p95 of 447 ms. One
route is fine. Twelve — which is what generating a loop costs — is five
seconds, and that is the budget gone.

These kernels are the same algorithms with the interpreter removed. Numba
needs plain typed arrays and no Python objects, so the priority queue is four
parallel arrays rather than `heapq`, and the graph arrives as the raw CSR
columns rather than a `GraphStore`.

`cache=True` writes the compiled code beside the module, so the two-to-ten
second compile is paid once per install rather than once per process. Warm
them at startup anyway — see `warm_up` — or the first request after a deploy
pays it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

from planner_core.core.units import EARTH_RADIUS_M

#: Marker for "no predecessor" in the search trees.
NO_ARC = -1


@njit(cache=True, nogil=True, inline="always")
def _swap(order: np.ndarray, keys: np.ndarray, left: int, right: int) -> None:
    """Exchange two heap entries."""
    keys[left], keys[right] = keys[right], keys[left]
    order[left], order[right] = order[right], order[left]


@njit(cache=True, nogil=True)
def _sift_up(keys: np.ndarray, order: np.ndarray, start: int) -> None:
    """Restore the heap property upward from `start`."""
    index = start
    while index > 0:
        parent = (index - 1) >> 1
        if keys[parent] <= keys[index]:
            break
        _swap(order, keys, parent, index)
        index = parent


@njit(cache=True, nogil=True)
def _sift_down(keys: np.ndarray, order: np.ndarray, size: int, start: int) -> None:
    """Restore the heap property downward from `start`."""
    index = start
    while True:
        left = 2 * index + 1
        right = left + 1
        smallest = index
        if left < size and keys[left] < keys[smallest]:
            smallest = left
        if right < size and keys[right] < keys[smallest]:
            smallest = right
        if smallest == index:
            break
        _swap(order, keys, smallest, index)
        index = smallest


@njit(cache=True, nogil=True, inline="always")
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    rlat1 = np.radians(lat1)
    rlat2 = np.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = np.radians(lon2 - lon1)
    inner = np.sin(dlat * 0.5) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon * 0.5) ** 2
    metres: float = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(inner))
    return metres


@njit(cache=True, nogil=True)
def astar_kernel(
    indptr: np.ndarray,
    arc_head: np.ndarray,
    arc_rev: np.ndarray,
    arc_allowed: np.ndarray,
    arc_cost: np.ndarray,
    node_lat: np.ndarray,
    node_lon: np.ndarray,
    source: int,
    target: int,
    heuristic_scale: float,
    max_settled: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """A* from `source` to `target`, returning the search tree.

    The predecessor array records the *arc* each node was reached by, not the
    node. That is what makes U-turn prohibition expressible: without it there
    is no way to know which way the rider was facing on arrival.

    `max_settled` of zero means no limit.
    """
    n_nodes = node_lat.shape[0]
    capacity = arc_head.shape[0] + 1

    dist = np.full(n_nodes, np.inf)
    pred_arc = np.full(n_nodes, NO_ARC, np.int32)
    pred_node = np.full(n_nodes, NO_ARC, np.int32)

    keys = np.empty(capacity)
    # Each heap slot holds a node and the arc it was reached by, packed so the
    # queue stays flat arrays rather than a list of tuples.
    order_node = np.empty(capacity, np.int32)
    order_arc = np.empty(capacity, np.int32)
    order_cost = np.empty(capacity)
    order = np.arange(capacity, dtype=np.int32)

    target_lat = node_lat[target]
    target_lon = node_lon[target]

    dist[source] = 0.0
    keys[0] = (
        _haversine_m(node_lat[source], node_lon[source], target_lat, target_lon) * heuristic_scale
    )
    order_node[0] = source
    order_arc[0] = NO_ARC
    order_cost[0] = 0.0
    size = 1

    settled = 0
    reached = 0

    while size > 0:
        slot = order[0]
        node = order_node[slot]
        arrived_on = order_arc[slot]
        cost = order_cost[slot]

        size -= 1
        if size > 0:
            _swap(order, keys, 0, size)
            _sift_down(keys, order, size, 0)

        if cost > dist[node]:
            continue
        if node == target:
            reached = 1
            break

        settled += 1
        if max_settled > 0 and settled > max_settled:
            break

        for arc in range(indptr[node], indptr[node + 1]):
            if arc_allowed[arc] == 0:
                continue
            if arrived_on != NO_ARC and arc == arc_rev[arrived_on]:
                continue

            head = arc_head[arc]
            candidate = cost + arc_cost[arc]
            if candidate < dist[head]:
                dist[head] = candidate
                pred_arc[head] = arc
                pred_node[head] = node

                free = order[size]
                keys[size] = (
                    candidate
                    + _haversine_m(node_lat[head], node_lon[head], target_lat, target_lon)
                    * heuristic_scale
                )
                order_node[free] = head
                order_arc[free] = arc
                order_cost[free] = candidate
                size += 1
                _sift_up(keys, order, size - 1)

    return dist, pred_arc, pred_node, settled, reached


@njit(cache=True, nogil=True)
def dijkstra_bounded_kernel(
    indptr: np.ndarray,
    arc_head: np.ndarray,
    arc_rev: np.ndarray,
    arc_allowed: np.ndarray,
    arc_cost: np.ndarray,
    arc_time: np.ndarray,
    source: int,
    cost_bound: float,
    time_bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Every node reachable from `source` within a cost and time bound.

    The loop generator's foundation: one search yields thousands of candidate
    turning points. Cost and time are tracked separately — the queue orders on
    cost, the budget test reads time — which is what keeps a reported ride
    duration honest.

    A bound of zero or less means unbounded on that axis.
    """
    n_nodes = indptr.shape[0] - 1
    capacity = arc_head.shape[0] + 1

    dist = np.full(n_nodes, np.inf)
    elapsed = np.full(n_nodes, np.inf)
    pred_arc = np.full(n_nodes, NO_ARC, np.int32)
    pred_node = np.full(n_nodes, NO_ARC, np.int32)

    keys = np.empty(capacity)
    order_node = np.empty(capacity, np.int32)
    order_arc = np.empty(capacity, np.int32)
    order_time = np.empty(capacity)
    order = np.arange(capacity, dtype=np.int32)

    dist[source] = 0.0
    elapsed[source] = 0.0
    keys[0] = 0.0
    order_node[0] = source
    order_arc[0] = NO_ARC
    order_time[0] = 0.0
    size = 1
    settled = 0

    while size > 0:
        slot = order[0]
        node = order_node[slot]
        arrived_on = order_arc[slot]
        cost = keys[0]
        spent = order_time[slot]

        size -= 1
        if size > 0:
            _swap(order, keys, 0, size)
            _sift_down(keys, order, size, 0)

        if cost > dist[node]:
            continue
        # The queue is cost-ordered, so once past the bound nothing later can
        # be inside it.
        if cost_bound > 0.0 and cost > cost_bound:
            break

        settled += 1

        for arc in range(indptr[node], indptr[node + 1]):
            if arc_allowed[arc] == 0:
                continue
            if arrived_on != NO_ARC and arc == arc_rev[arrived_on]:
                continue

            head = arc_head[arc]
            candidate = cost + arc_cost[arc]
            candidate_time = spent + arc_time[arc]
            if cost_bound > 0.0 and candidate > cost_bound:
                continue
            if time_bound > 0.0 and candidate_time > time_bound:
                continue

            if candidate < dist[head]:
                dist[head] = candidate
                elapsed[head] = candidate_time
                pred_arc[head] = arc
                pred_node[head] = node

                free = order[size]
                keys[size] = candidate
                order_node[free] = head
                order_arc[free] = arc
                order_time[free] = candidate_time
                size += 1
                _sift_up(keys, order, size - 1)

    return dist, elapsed, pred_arc, pred_node, settled


def warm_up() -> None:
    """Compile the kernels against a trivial graph.

    Numba's first call pays two to ten seconds of compilation. Without this at
    startup, the first real request after every deploy wears it — and with a
    request timeout in place, that request fails.
    """
    indptr = np.array([0, 1, 2], dtype=np.int32)
    arc_head = np.array([1, 0], dtype=np.int32)
    arc_rev = np.array([1, 0], dtype=np.int32)
    allowed = np.ones(2, dtype=np.uint8)
    cost = np.ones(2, dtype=np.float64)
    lat = np.array([50.0, 50.001])
    lon = np.array([5.0, 5.001])

    astar_kernel(indptr, arc_head, arc_rev, allowed, cost, lat, lon, 0, 1, 1.0, 0)
    dijkstra_bounded_kernel(indptr, arc_head, arc_rev, allowed, cost, cost, 0, 0.0, 0.0)


def reconstruct(
    pred_arc: npt.NDArray[np.int32],
    pred_node: npt.NDArray[np.int32],
    source: int,
    target: int,
) -> tuple[int, ...]:
    """Walk a search tree back from `target` to `source`."""
    arcs: list[int] = []
    node = target
    while node != source:
        arc = int(pred_arc[node])
        if arc == NO_ARC:
            msg = "Predecessor chain broke before reaching the source"
            raise RuntimeError(msg)
        arcs.append(arc)
        node = int(pred_node[node])
    arcs.reverse()
    return tuple(arcs)
