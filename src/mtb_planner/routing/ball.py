"""The subgraph one request is allowed to search.

A loop that fits a two-hour budget cannot leave a circle of about 30 km around
the start, so searching the whole of Belgium to find it is waste. Restricting to
that circle first is what makes the rest of the generator affordable: on the
full area of interest it is the difference between 11M arcs and a few hundred
thousand, and every one of the dozen searches a loop request makes pays it.

The ball is expressed as **masks over the graph's own indices**, not as a
renumbered subgraph. Renumbering would mean translating every node and arc on
the way in and out, and one missed translation produces a route that is subtly
in the wrong place rather than an error. Masks cost one byte per arc and keep
every index directly comparable with the full graph.

Distances are measured on an equirectangular projection about the centre. Over
30 km that is accurate to a few metres — far finer than needed to decide
whether a node is worth searching, and it lets the whole graph be tested with
two multiplications instead of a haversine per node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mtb_planner.core.units import EARTH_RADIUS_M
from mtb_planner.graph.store import GraphStore

#: Metres per degree of latitude.
_M_PER_DEG = EARTH_RADIUS_M * math.pi / 180.0

#: How much of the time budget a loop can spend travelling away from the start
#: before it has to come back. Half, by definition of a loop — the radius that
#: follows is generous, because it assumes the rider goes in a straight line at
#: the fastest speed the model allows.
OUTBOUND_FRACTION = 0.5

#: Never search a ball smaller than this, whatever the budget. A fifteen-minute
#: request still needs somewhere to go.
MIN_RADIUS_M = 2_000.0


def loop_radius_m(budget_s: float, max_speed_mps: float) -> float:
    """How far from the start a loop within `budget_s` could possibly reach."""
    return max(MIN_RADIUS_M, OUTBOUND_FRACTION * budget_s * max_speed_mps)


@dataclass(frozen=True)
class Ball:
    """The nodes and arcs a request may use, as masks over the whole graph."""

    centre: int
    radius_m: float
    node_inside: npt.NDArray[np.bool_]
    arc_inside: npt.NDArray[np.bool_]

    @property
    def n_nodes(self) -> int:
        """Nodes inside the ball."""
        return int(self.node_inside.sum())

    @property
    def n_arcs(self) -> int:
        """Arcs inside the ball."""
        return int(self.arc_inside.sum())

    def contains_node(self, node: int) -> bool:
        """Whether a node is inside."""
        return bool(self.node_inside[node])


def arc_tails(store: GraphStore) -> npt.NDArray[np.int32]:
    """The tail node of every arc, vectorised.

    The CSR layout gives heads directly but tails only implicitly, as the row
    each arc sits in.
    """
    counts = np.diff(store.indptr).astype(np.int64)
    return np.repeat(np.arange(store.n_nodes, dtype=np.int32), counts)


def ball_around(store: GraphStore, centre: int, radius_m: float) -> Ball:
    """The subgraph within `radius_m` of `centre`.

    An arc is inside only when both its endpoints are, so a search confined to
    the ball can never step outside it.
    """
    lat0 = float(store.node_lat[centre])
    lon0 = float(store.node_lon[centre])
    lon_scale = _M_PER_DEG * math.cos(math.radians(lat0))

    north = (store.node_lat - lat0) * _M_PER_DEG
    east = (store.node_lon - lon0) * lon_scale
    node_inside = (north * north + east * east) <= radius_m * radius_m

    arc_inside = node_inside[arc_tails(store)] & node_inside[store.arc_head]

    return Ball(
        centre=centre,
        radius_m=radius_m,
        node_inside=node_inside,
        arc_inside=arc_inside,
    )


def edges_inside(store: GraphStore, ball: Ball) -> npt.NDArray[np.bool_]:
    """Which edges have at least one arc inside the ball.

    Used to restrict the cost model's work. An edge is included if either
    direction is usable, because the two directions share one time calculation.
    """
    inside = np.zeros(store.n_edges, dtype=np.bool_)
    inside[store.arc_edge[ball.arc_inside]] = True
    return inside
