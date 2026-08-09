"""Making a corridor expensive so the return leg finds another one.

A loop built from two shortest paths between the same pair of points is not a
loop — it is an out-and-back, because the second search finds the same trail as
the first. Multiplying the cost of the arcs already used, and of their reverse
twins, is what forces the return somewhere else.

The factor matters more than it looks. Too small and the return leg still comes
back down the outbound trail, because a small penalty on a good trail still
beats a detour. Too large and a valley with genuinely one trail through it
provokes a twenty-kilometre diversion rather than an honest partial overlap.
Four is enough to prefer any real alternative without inventing one.

**Only `cost` is touched.** Time and distance are what the rider is told, and a
penalty is a routing preference, not a claim about how long the ride takes. If
a penalty ever reached `time_s`, every reported duration on a loop would be
inflated by the amount of backtracking the generator happened to discourage.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from mtb_planner.cost.costvector import ArcCosts
from mtb_planner.graph.store import GraphStore

#: How much dearer an already-ridden arc becomes on the return leg.
RETURN_PENALTY = 4.0


def penalise(
    costs: ArcCosts,
    store: GraphStore,
    arcs: Iterable[int],
    factor: float = RETURN_PENALTY,
    forbid: Iterable[int] = (),
) -> ArcCosts:
    """Return a cost vector that discourages `arcs` and their reverse twins.

    `forbid` removes arcs outright, which is how the join at a loop's turning
    point is stopped from being an immediate U-turn: the search kernels prohibit
    doubling back *within* one search, but the seam between the outbound and
    return legs is nobody's business but the caller's.

    Raising costs never breaks A* admissibility — the heuristic is scaled by the
    fastest second per metre in the graph, which is derived from `time_s` and
    `distance_m`, and neither is touched here.
    """
    index = np.fromiter(arcs, dtype=np.intp)
    cost = costs.cost.copy()
    if index.size:
        both = np.concatenate((index, store.arc_rev[index].astype(np.intp)))
        cost[both] *= factor

    banned = np.fromiter(forbid, dtype=np.intp)
    allowed = costs.allowed
    if banned.size:
        allowed = costs.allowed.copy()
        allowed[banned] = 0

    return ArcCosts(
        cost=cost,
        time_s=costs.time_s,
        distance_m=costs.distance_m,
        allowed=allowed,
        preference=costs.preference,
    )
