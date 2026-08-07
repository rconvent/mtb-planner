"""Turning preference into a searchable cost.

Three independent scalars per arc, and keeping them independent is the whole
design:

    t(a)  seconds   pure physics, no preference term, ever
    d(a)  metres    pure geometry
    p(a)  [-1, +1]  pure preference, independent of length

    cost(a) = t(a) · (1 + λ · (1 − p_norm(a)))

Three properties follow, and all three are load-bearing:

1. `cost >= t`, so straight-line distance over the maximum plausible speed is
   an admissible A* heuristic. Correct A* comes free.
2. `cost <= (1 + λ)·t`, which bounds how far a budget-limited search must
   look — the next phase's loop generator depends on it.
3. Budget accounting never reads `cost`. Labels carry cost, time and
   distance separately; the queue orders on cost while every budget test
   reads time. The "26 km, two hours" shown to a rider is measured, not
   inferred from the thing that drove the routing.

The tempting alternative — `cost = t + penalty_seconds` — breaks all three.
Additive penalties corrupt the budget and destroy heuristic admissibility the
moment a penalty can be negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mtb_planner.cost.profiles import FEATURE_NAMES, RideProfile
from mtb_planner.cost.speed import Rider, edge_times, speed_table
from mtb_planner.features.enums import PAVED_SURFACES, TechConfidence, WayClass
from mtb_planner.features.model import EdgeFeatures
from mtb_planner.graph.store import GraphStore

#: Technicality this far above a profile's ceiling is excluded outright. Never
#: route a rider down something they have said they cannot ride.
TECH_CEILING_MARGIN = 1.0

#: Grade at which the climb and descent features saturate.
_CLIMB_REFERENCE = 0.08
_DESCENT_REFERENCE = 0.10


@dataclass(frozen=True)
class ArcCosts:
    """Per-arc cost, time and distance, ready for a search."""

    cost: npt.NDArray[np.float64]
    time_s: npt.NDArray[np.float64]
    distance_m: npt.NDArray[np.float64]
    allowed: npt.NDArray[np.uint8]
    preference: npt.NDArray[np.float64]
    """How much the rider wants each arc, from -1 to +1.

    Kept alongside the cost rather than folded into it, because the loop
    generator needs it as a quantity in its own right: to rank pivots by
    preference density, to choose which stretch of an over-long route to
    replace, and to score candidates. Recomputing it per candidate would mean
    a second copy of the feature weighting, free to drift from this one.
    """

    def __len__(self) -> int:
        """Number of arcs described."""
        return int(self.cost.shape[0])

    @property
    def min_seconds_per_metre(self) -> float:
        """Fastest any arc is traversed, for scaling the A* heuristic.

        The heuristic must never overestimate, so it uses the quickest metre
        anywhere in the graph rather than an average.
        """
        moving = self.distance_m > 0.0
        if not moving.any():
            return 0.0
        return float(np.min(self.time_s[moving] / self.distance_m[moving]))


def _feature_matrix(
    features: EdgeFeatures, elevation_grade: npt.NDArray[np.float64], profile: RideProfile
) -> npt.NDArray[np.float64]:
    """Assemble the normalised features a profile weights, one column each."""
    columns: dict[str, npt.NDArray[np.float64]] = {}

    # Technicality is a band match, not "more is better": an XC rider wants an
    # S1 and is no happier on an S4 than on a towpath.
    offset = features.tech.astype(np.float64) - profile.tech_target
    columns["tech"] = np.exp(-(offset**2) / (2.0 * profile.tech_sigma**2))

    columns["singletrack"] = features.singletrack.astype(np.float64)
    columns["signed_mtb"] = np.clip(features.n_mtb_routes.astype(np.float64) / 3.0, 0.0, 1.0)
    columns["flow"] = features.flow.astype(np.float64)

    paved = np.isin(features.surface_class, [int(s) for s in PAVED_SURFACES])
    columns["paved"] = paved.astype(np.float64)
    columns["unpaved"] = 1.0 - columns["paved"]
    columns["fireroad"] = (features.way_class == int(WayClass.FIREROAD)).astype(np.float64)
    columns["traffic"] = features.traffic.astype(np.float64)

    columns["descent"] = np.clip(-elevation_grade / _DESCENT_REFERENCE, 0.0, 1.0)
    columns["climb"] = np.clip(elevation_grade / _CLIMB_REFERENCE, 0.0, 1.0)

    # A mild aversion to edges nothing is known about, so a route through
    # guesswork loses to an equally good route through measured ground.
    columns["unknown"] = (features.tech_conf == int(TechConfidence.DEFAULT)).astype(np.float64)

    return np.column_stack([columns[name] for name in FEATURE_NAMES])


def preference(
    features: EdgeFeatures, elevation_grade: npt.NDArray[np.float64], profile: RideProfile
) -> npt.NDArray[np.float64]:
    """How much a rider wants each edge, from −1 to +1.

    `tanh` bounds the result and saturates gracefully when several features
    agree, so no single weight can dominate the rest.
    """
    matrix = _feature_matrix(features, elevation_grade, profile)
    return np.tanh(matrix @ np.asarray(profile.weights, dtype=np.float64))


def max_speed_mps(rider: Rider) -> float:
    """The fastest the time model will ever let this rider travel.

    Graph-independent, which is what lets the search ball be sized before any
    cost vector exists — the ball is needed to build the cost vector cheaply,
    so the two cannot depend on each other.
    """
    return float(speed_table(rider).max())


def materialise(
    store: GraphStore,
    profile: RideProfile,
    rider: Rider | None = None,
    arc_mask: npt.NDArray[np.bool_] | None = None,
) -> ArcCosts:
    """Build the per-arc cost, time and distance a search needs.

    `arc_mask` restricts the result to a subgraph: arcs outside it are marked
    unusable, and the physics is solved only for the edges that remain. On a
    country-sized graph that is the difference between roughly 120 ms and a few
    milliseconds, and a loop request cannot afford the former repeatedly.
    """
    rider = rider or Rider()
    lengths = store.edge_length_m

    # Which edges the physics has to be solved for. Everything else keeps its
    # zero-initialised placeholder and is unreachable via `allowed`.
    if arc_mask is None:
        selected = np.arange(store.n_edges, dtype=np.intp)
    else:
        inside = np.zeros(store.n_edges, dtype=np.bool_)
        inside[store.arc_edge[arc_mask]] = True
        selected = np.flatnonzero(inside)

    features = store.features.select(selected)
    sub_lengths = lengths[selected]

    sub_forward_s, sub_backward_s = edge_times(
        grade_hist=store.elevation.grade_hist[selected],
        surface_class=features.surface_class,
        tech=features.tech,
        tech_up=features.tech_up,
        access=store.edge_access[selected],
        lengths=sub_lengths,
        table=speed_table(rider),
    )

    # Mean grade along the edge, signed in its stored direction. Only used to
    # score preference; time comes from the full histogram.
    rise = store.elevation.ascent_m[selected].astype(np.float64) - store.elevation.descent_m[
        selected
    ].astype(np.float64)
    grade = np.divide(rise, sub_lengths, out=np.zeros_like(sub_lengths), where=sub_lengths > 0)

    # Scattered back to full length so arc lookups stay plain indexing rather
    # than a translation table.
    forward_s = np.zeros(store.n_edges, dtype=np.float64)
    backward_s = np.zeros(store.n_edges, dtype=np.float64)
    forward_p = np.zeros(store.n_edges, dtype=np.float64)
    backward_p = np.zeros(store.n_edges, dtype=np.float64)
    tech_full = np.zeros(store.n_edges, dtype=np.float64)

    forward_s[selected] = sub_forward_s
    backward_s[selected] = sub_backward_s
    forward_p[selected] = preference(features, grade, profile)
    backward_p[selected] = preference(features, -grade, profile)
    tech_full[selected] = features.tech

    is_forward = store.arc_forward.astype(np.bool_)
    edge_of_arc = store.arc_edge

    time_s = np.where(is_forward, forward_s[edge_of_arc], backward_s[edge_of_arc])
    pref = np.where(is_forward, forward_p[edge_of_arc], backward_p[edge_of_arc])
    distance = lengths[edge_of_arc]

    normalised = (pref + 1.0) / 2.0
    cost = time_s * (1.0 + profile.detour * (1.0 - normalised))

    # Anything beyond the rider's declared ceiling is removed, not penalised.
    too_hard = tech_full[edge_of_arc] > profile.tech_max + TECH_CEILING_MARGIN
    allowed = np.where(too_hard, 0, store.arc_allowed).astype(np.uint8)
    if arc_mask is not None:
        allowed = np.where(arc_mask, allowed, 0).astype(np.uint8)

    return ArcCosts(
        cost=cost,
        time_s=time_s,
        distance_m=distance,
        allowed=allowed,
        preference=pref,
    )
