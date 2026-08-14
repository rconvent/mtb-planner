"""Measuring, scoring and diversifying candidate loops.

The generator produces far more candidates than a rider wants to see, and the
ones it produces are not equally good. Three jobs live here:

**Measure.** Everything ranking reads is summed from the arcs actually ridden —
never from the cost that selected them. That is the same discipline the cost
model keeps, and for the same reason: a figure derived from the preference that
drove the search cannot be used to judge the result of that search.

**Score.** A weighted sum over qualities a rider would recognise, with the
weights as data. Technicality is a band match rather than "more is better",
matching the cost function: a rider asking for flow trails is no happier on an
S4 than on a towpath.

**Diversify.** The top few by score are usually near-identical — the same trail
with a different corner cut. Maximal marginal relevance trades a little score
for genuine difference, which is what a rider perceives as being offered
choices rather than one route four times.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mtb_planner.cost.costvector import ArcCosts
from mtb_planner.cost.profiles import RideProfile
from mtb_planner.features.enums import PAVED_SURFACES, TechConfidence
from mtb_planner.graph.store import GraphStore

#: How much score a candidate gives up to be different from the ones already
#: chosen. Zero returns the top-scoring set, one ignores score entirely.
DIVERSITY = 0.35


@dataclass(frozen=True)
class ScoreWeights:
    """What makes one loop better than another.

    Data, not code: retuning these is a configuration decision, and keeping
    them in one frozen object means a change cannot be made in half the places.
    """

    preference: float = 0.30
    budget_fit: float = 0.25
    singletrack: float = 0.15
    signed: float = 0.10
    tech_fit: float = 0.20
    overlap: float = -0.25
    paved: float = -0.15
    traffic: float = -0.10
    unknown: float = -0.05


DEFAULT_WEIGHTS = ScoreWeights()


@dataclass(frozen=True)
class LoopMetrics:
    """What is known about one candidate loop, all measured from its arcs."""

    time_s: float
    distance_m: float
    overlap_ratio: float
    preference: float
    singletrack: float
    signed: float
    paved: float
    traffic: float
    unknown: float
    tech: float

    def budget_fit(self, budget_s: float) -> float:
        """How close this loop lands to the requested time, from 0 to 1."""
        if budget_s <= 0.0:
            return 0.0
        return float(max(0.0, 1.0 - abs(self.time_s - budget_s) / budget_s))

    def budget_error(self, budget_s: float) -> float:
        """Signed fractional miss against the budget."""
        if budget_s <= 0.0:
            return 0.0
        return (self.time_s - budget_s) / budget_s


def overlap_ratio(store: GraphStore, arcs: Sequence[int]) -> float:
    """Fraction of a route's length that is ridden more than once.

    Counted per edge, in either direction, so an out-and-back scores 0.5: half
    of the distance covered is a repeat of the other half. Some overlap is
    legitimate — a trailhead often has one way in — which is why this is
    reported and thresholded rather than forbidden.
    """
    if not arcs:
        return 0.0
    edges = store.arc_edge[np.asarray(arcs, dtype=np.intp)]
    unique, counts = np.unique(edges, return_counts=True)
    lengths = store.edge_length_m[unique]
    total = float((lengths * counts).sum())
    if total <= 0.0:
        return 0.0
    return float((lengths * (counts - 1)).sum() / total)


def _length_weighted(values: npt.NDArray[np.float64], lengths: npt.NDArray[np.float64]) -> float:
    """Mean of `values` weighted by length, or zero for an empty route."""
    total = float(lengths.sum())
    if total <= 0.0:
        return 0.0
    return float((values * lengths).sum() / total)


def measure(store: GraphStore, costs: ArcCosts, arcs: Sequence[int]) -> LoopMetrics:
    """Summarise a candidate from the arcs it actually rides."""
    if not arcs:
        return LoopMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    index = np.asarray(arcs, dtype=np.intp)
    edges = store.arc_edge[index].astype(np.intp)
    lengths = costs.distance_m[index]
    features = store.features

    paved = np.isin(features.surface_class[edges], [int(s) for s in PAVED_SURFACES])
    unknown = features.tech_conf[edges] == int(TechConfidence.DEFAULT)

    return LoopMetrics(
        time_s=float(costs.time_s[index].sum()),
        distance_m=float(lengths.sum()),
        overlap_ratio=overlap_ratio(store, arcs),
        preference=_length_weighted(costs.preference[index], lengths),
        singletrack=_length_weighted(features.singletrack[edges].astype(np.float64), lengths),
        signed=_length_weighted(features.signed_mtb[edges].astype(np.float64), lengths),
        paved=_length_weighted(paved.astype(np.float64), lengths),
        traffic=_length_weighted(features.traffic[edges].astype(np.float64), lengths),
        unknown=_length_weighted(unknown.astype(np.float64), lengths),
        tech=_length_weighted(features.tech[edges].astype(np.float64), lengths),
    )


def score(
    metrics: LoopMetrics,
    profile: RideProfile,
    budget_s: float,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> float:
    """How good this loop is for this rider, higher being better.

    Preference is mapped from [−1, +1] onto [0, 1] so every term shares a scale
    and no weight has to compensate for a different range.
    """
    offset = metrics.tech - profile.tech_target
    tech_fit = float(np.exp(-(offset**2) / (2.0 * profile.tech_sigma**2)))

    return (
        weights.preference * (metrics.preference + 1.0) / 2.0
        + weights.budget_fit * metrics.budget_fit(budget_s)
        + weights.singletrack * metrics.singletrack
        + weights.signed * metrics.signed
        + weights.tech_fit * tech_fit
        + weights.overlap * metrics.overlap_ratio
        + weights.paved * metrics.paved
        + weights.traffic * metrics.traffic
        + weights.unknown * metrics.unknown
    )


def similarity(store: GraphStore, left: Sequence[int], right: Sequence[int]) -> float:
    """How much ground two routes share, from 0 to 1.

    Jaccard over edges rather than arcs, so two loops riding the same trail in
    opposite directions count as the same trail — which is how they look to a
    rider deciding between them.
    """
    if not left or not right:
        return 0.0
    a = set(store.arc_edge[np.asarray(left, dtype=np.intp)].tolist())
    b = set(store.arc_edge[np.asarray(right, dtype=np.intp)].tolist())
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def diversify(
    store: GraphStore,
    routes: Sequence[Sequence[int]],
    scores: Sequence[float],
    count: int,
    gamma: float = DIVERSITY,
) -> tuple[int, ...]:
    """Pick `count` candidates that are both good and unalike.

    Returns indices into `routes`, best first. The first pick is always the
    top-scoring candidate; each one after it trades `gamma` of score against how
    much ground it shares with what is already chosen.

    Scores are used raw, not rescaled across the candidate set. Min-max
    normalisation is the textbook move here and it is wrong at this scale: over
    a handful of candidates it stretches whatever the score spread happens to be
    onto the full range, so a hundredth of a point becomes a decisive advantage
    and near-duplicates always win. `score` already returns a roughly [0, 1]
    quantity, which is the same scale as the similarity it is traded against.

    Ties are broken by the candidate's own order, so the result cannot wobble
    between identical requests.
    """
    if not routes:
        return ()

    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="stable")[::-1]

    chosen: list[int] = [int(order[0])]
    remaining = [int(i) for i in order[1:]]

    while remaining and len(chosen) < count:
        best_index = 0
        best_value = -np.inf
        for position, candidate in enumerate(remaining):
            overlap = max(similarity(store, routes[candidate], routes[picked]) for picked in chosen)
            value = (1.0 - gamma) * scores[candidate] - gamma * overlap
            if value > best_value:
                best_value = value
                best_index = position
        chosen.append(remaining.pop(best_index))

    return tuple(chosen)
