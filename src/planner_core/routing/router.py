"""The one way to search this graph.

Every search — point to point today, loop generation next — goes through this
facade. The alternative is each caller reaching for the kernels directly and
quietly growing its own notion of what a U-turn is, or which arcs a profile
forbids. Keeping that in one place is what stops the loop generator becoming a
second router.

A `Router` binds a graph to a materialised cost vector, so the expensive part
of a request happens once and every search after it is a kernel call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from planner_core.config.logging import get_logger
from planner_core.cost.costvector import ArcCosts
from planner_core.graph.store import GraphStore
from planner_core.routing.budget import INITIAL_SETTLE_CEILING, SETTLE_GROWTH, Deadline
from planner_core.routing.kernels import astar_kernel, dijkstra_bounded_kernel, reconstruct

logger = get_logger(__name__)

#: Passed to the kernels to mean "no limit"; they take zero rather than an
#: optional, because Numba has no `None`.
UNBOUNDED = 0.0


@dataclass(frozen=True)
class SearchResult:
    """A completed point-to-point search."""

    arcs: tuple[int, ...]
    cost: float
    settled: int
    reached: bool
    truncated: bool = False
    """Whether a settle ceiling stopped the search before it could finish.

    Distinguishes "there is no route" from "we ran out of budget looking",
    which need different answers: the first is final, the second is worth
    retrying with more room.
    """


@dataclass(frozen=True)
class Frontier:
    """Everything a bounded search reached, and how it got there.

    The loop generator picks its turning points out of this: one search, then
    thousands of candidate pivots, rather than one search per candidate.
    """

    cost: npt.NDArray[np.float64]
    time_s: npt.NDArray[np.float64]
    pred_arc: npt.NDArray[np.int32]
    pred_node: npt.NDArray[np.int32]
    source: int
    settled: int

    def reached(self) -> npt.NDArray[np.intp]:
        """Nodes the search actually settled."""
        return np.flatnonzero(np.isfinite(self.cost))

    def within_time(self, low_s: float, high_s: float) -> npt.NDArray[np.intp]:
        """Nodes whose travel time falls in a band.

        Time, never cost: a pivot half a ride away has to be half a *ride*
        away, not half a preference-weighted cost away.
        """
        return np.flatnonzero((self.time_s >= low_s) & (self.time_s <= high_s))

    def path_to(self, node: int) -> tuple[int, ...]:
        """The arc sequence from the source to a reached node."""
        return reconstruct(self.pred_arc, self.pred_node, self.source, node)


class Router:
    """Searches one graph under one cost model."""

    def __init__(self, store: GraphStore, costs: ArcCosts) -> None:
        self._store = store
        self._costs = costs

    @property
    def costs(self) -> ArcCosts:
        """The cost vector every search here uses."""
        return self._costs

    def shortest(self, source: int, target: int, max_settled: int | None = None) -> SearchResult:
        """Least-cost route between two nodes."""
        if source == target:
            return SearchResult(arcs=(), cost=0.0, settled=0, reached=True)

        store, costs = self._store, self._costs
        dist, pred_arc, pred_node, settled, reached = astar_kernel(
            store.indptr,
            store.arc_head,
            store.arc_rev,
            costs.allowed,
            costs.cost,
            store.node_lat,
            store.node_lon,
            source,
            target,
            costs.min_seconds_per_metre,
            0 if max_settled is None else max_settled,
        )

        # The kernel settles one node past its ceiling before breaking, so a
        # count above the ceiling means it stopped early rather than ran out of
        # graph to explore.
        truncated = max_settled is not None and max_settled > 0 and settled > max_settled

        if not reached:
            return SearchResult(
                arcs=(),
                cost=float("inf"),
                settled=settled,
                reached=False,
                truncated=truncated,
            )

        return SearchResult(
            arcs=reconstruct(pred_arc, pred_node, source, target),
            cost=float(dist[target]),
            settled=settled,
            reached=True,
        )

    def shortest_within(
        self,
        source: int,
        target: int,
        deadline: Deadline | None = None,
        initial_ceiling: int | None = None,
    ) -> SearchResult:
        """Least-cost route, abandoned if the time budget runs out.

        Searches under a settle ceiling and raises the ceiling until the target
        is reached, the graph is exhausted, or the budget is spent. See
        `budget.py` for why the clock is enforced out here rather than inside
        the kernel.
        """
        if deadline is None:
            return self.shortest(source, target)

        # Read here rather than bound as a default, so the constant stays a
        # single source of truth instead of being frozen at import time.
        ceiling = max(1, INITIAL_SETTLE_CEILING if initial_ceiling is None else initial_ceiling)
        while True:
            result = self.shortest(source, target, max_settled=ceiling)
            if result.reached or not result.truncated:
                # Reached it, or ran out of reachable graph — either way the
                # answer is final and more budget would not change it.
                return result
            if deadline.expired:
                logger.info(
                    "Search from %d to %d abandoned after %d settled nodes: budget spent",
                    source,
                    target,
                    result.settled,
                )
                return result
            ceiling *= SETTLE_GROWTH

    def reachable_within(
        self,
        source: int,
        cost_bound: float | None = None,
        time_bound: float | None = None,
    ) -> Frontier:
        """Everything reachable from `source` inside the given bounds.

        With cost bounded and `cost <= (1 + λ)·time` guaranteed by the cost
        construction, a time budget of `T` is fully covered by a cost bound of
        `(1 + λ)·T` — which is what makes this search provably complete rather
        than merely usually adequate.
        """
        store, costs = self._store, self._costs
        cost, elapsed, pred_arc, pred_node, settled = dijkstra_bounded_kernel(
            store.indptr,
            store.arc_head,
            store.arc_rev,
            costs.allowed,
            costs.cost,
            costs.time_s,
            source,
            UNBOUNDED if cost_bound is None else cost_bound,
            UNBOUNDED if time_bound is None else time_bound,
        )
        return Frontier(
            cost=cost,
            time_s=elapsed,
            pred_arc=pred_arc,
            pred_node=pred_node,
            source=source,
            settled=settled,
        )

    def time_of(self, arcs: tuple[int, ...]) -> float:
        """Riding time for an arc sequence, in seconds.

        Summed from the arcs, never from the cost that selected them.
        """
        return float(sum(self._costs.time_s[arc] for arc in arcs))

    def distance_of(self, arcs: tuple[int, ...]) -> float:
        """Ground covered by an arc sequence, in metres."""
        return float(sum(self._costs.distance_m[arc] for arc in arcs))
