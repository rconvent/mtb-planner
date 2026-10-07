"""Spreading surveyed difficulty across the trails it belongs to.

Worth more than any amount of tuning on the estimation table, because real
networks are tagged in clusters: one mapper surveys a whole trail centre, and
the neighbouring kilometre of the same trail gets nothing. Where enough of a
trail is surveyed, that knowledge should inform the rest of it.

Trails are grouped three ways: shared name, shared `route=mtb` relation, and
corridor adjacency between trail edges of matching character.

The first two are statements of intent by a mapper, and they are the obvious
choice — but measured against Luxembourg they are nearly worthless. Of 4,590
surveyed segments only 111 also carry a name, and just 16 of 4,061 name and
route groups contain any surveyed member at all. `mtb:scale` in this data sits
on unnamed forest paths, so the clustering is real but spatial rather than
nominal.

Adjacency is what reaches it. The risk is obvious — blindly chaining
neighbours would merge a technical descent into the fire road it joins — so
edges are only linked when both are trail-class and share a surface class.
That keeps a group to things that plausibly ride alike.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from planner_core.config.logging import get_logger
from planner_core.features.enums import TechConfidence, WayClass
from planner_core.features.model import EdgeFeatures
from planner_core.graph.segments import Segment

logger = get_logger(__name__)

#: Way classes that ride like trails, and so may be chained into a corridor.
#: Fire roads and anything sealed are excluded: they are what a trail runs
#: into, not part of it.
_TRAIL_WAY_CLASSES: frozenset[int] = frozenset(
    {int(WayClass.SINGLETRACK), int(WayClass.DOUBLETRACK)}
)

#: A group needs this share of its length surveyed before its median is
#: trusted enough to inform the rest.
MIN_SURVEYED_FRACTION = 0.20

#: How much of a propagated value comes from the group rather than the edge's
#: own estimate. The estimate still carries real information — surface and
#: width are measured on that edge — so it is blended, not overwritten.
GROUP_WEIGHT = 0.6


class _DisjointSet:
    """Union-find over segment indices."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        """Representative of an item's group, with path compression."""
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        """Merge two groups."""
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root

    def groups(self) -> dict[int, list[int]]:
        """Every group with more than one member."""
        collected: dict[int, list[int]] = defaultdict(list)
        for item in range(len(self._parent)):
            collected[self.find(item)].append(item)
        return {root: members for root, members in collected.items() if len(members) > 1}


def _link_by_corridor(
    segments: Sequence[Segment], features: EdgeFeatures, disjoint: _DisjointSet
) -> None:
    """Join trail edges that meet and share a surface class.

    The character test is what makes this safe. Without it a chain would run
    from a hand-cut descent, through the fire road it lands on, and out along
    a farm track — carrying a surveyed S3 with it the whole way.
    """
    by_node: dict[int, list[int]] = defaultdict(list)
    trail = np.isin(features.way_class, list(_TRAIL_WAY_CLASSES))

    for index, segment in enumerate(segments):
        if trail[index]:
            by_node[segment.u].append(index)
            by_node[segment.v].append(index)

    for touching in by_node.values():
        if len(touching) < 2:
            continue
        first = touching[0]
        for other in touching[1:]:
            if features.surface_class[first] == features.surface_class[other]:
                disjoint.union(first, other)


def _build_groups(segments: Sequence[Segment], features: EdgeFeatures) -> dict[int, list[int]]:
    """Group segments that plausibly belong to the same trail."""
    disjoint = _DisjointSet(len(segments))
    by_name: dict[str, int] = {}
    by_route: dict[int, int] = {}

    for index, segment in enumerate(segments):
        name = (segment.tags.name or "").strip().lower()
        if name:
            disjoint.union(by_name.setdefault(name, index), index)
        for route_id in segment.routes:
            disjoint.union(by_route.setdefault(route_id, index), index)

    _link_by_corridor(segments, features, disjoint)
    return disjoint.groups()


def _length_weighted_median(
    values: npt.NDArray[np.float32], weights: npt.NDArray[np.float64]
) -> float:
    """Median of `values`, weighting each by its length.

    A plain median would let a hundred metres of connector outvote a kilometre
    of the trail the group is named after.
    """
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    half = cumulative[-1] / 2.0
    return float(values[order][int(np.searchsorted(cumulative, half))])


def propagate_difficulty(
    features: EdgeFeatures,
    segments: Sequence[Segment],
    lengths: npt.NDArray[np.float64],
    min_surveyed_fraction: float = MIN_SURVEYED_FRACTION,
    group_weight: float = GROUP_WEIGHT,
) -> EdgeFeatures:
    """Blend surveyed difficulty into the unsurveyed parts of the same trail."""
    tech = features.tech.copy()
    confidence = features.tech_conf.copy()

    surveyed = features.tech_conf == int(TechConfidence.SURVEYED)
    if not surveyed.any():
        logger.info("Propagation: nothing surveyed, leaving every estimate as it is")
        return features.with_difficulty(tech, confidence)

    improved = 0
    groups = _build_groups(segments, features)
    for members in groups.values():
        index = np.asarray(members, dtype=np.intp)
        total = float(lengths[index].sum())
        if total <= 0.0:
            continue

        known = index[surveyed[index]]
        if known.size == 0 or float(lengths[known].sum()) / total < min_surveyed_fraction:
            continue

        # Only edges that know less than a proxy are worth overwriting.
        targets = index[confidence[index] <= int(TechConfidence.ESTIMATED)]
        if targets.size == 0:
            continue

        median = _length_weighted_median(features.tech[known], lengths[known])
        tech[targets] = group_weight * median + (1.0 - group_weight) * features.tech[targets]
        confidence[targets] = int(TechConfidence.PROXY)
        improved += int(targets.size)

    logger.info(
        "Propagation: %d group(s), %d edge(s) raised to proxy confidence",
        len(groups),
        improved,
    )
    return features.with_difficulty(tech, confidence)


def confidence_coverage(
    features: EdgeFeatures, lengths: npt.NDArray[np.float64]
) -> dict[str, float]:
    """Share of network length at each confidence level.

    The headline number for this phase: how much of the network the difficulty
    model actually knows something about, measured in metres rather than edge
    counts so a thousand short urban links cannot flatter it.
    """
    total = float(lengths.sum())
    if total <= 0.0:
        return {level.name.lower(): 0.0 for level in TechConfidence}

    return {
        level.name.lower(): float(lengths[features.tech_conf == int(level)].sum()) / total
        for level in TechConfidence
    }
