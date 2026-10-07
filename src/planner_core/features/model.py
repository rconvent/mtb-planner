"""The per-edge feature vector.

Columnar, like everything else that scales with edge count: the cost function
materialises weights over a few hundred thousand arcs per request, and that
has to be a vectorised operation rather than a loop over objects.

Deliberately stored as explicit arrays rather than packed into a bitfield.
Packing would save around 100 MB across a full build, but `tech` is a
continuous value after propagation blends surveyed neighbours into estimates,
so it cannot be recovered from quantised categoricals. Storing what the model
actually produces is worth the space.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from planner_core.features.classify import classify_way, singletrack_score, traffic_exposure
from planner_core.features.enums import TechConfidence
from planner_core.features.surface import classify_surface, rollability
from planner_core.graph.segments import Segment
from planner_core.ingest.pbf_reader import RouteRelation
from planner_core.sport import Sport, active_sport

#: `route_flags` bits. A signed route is one waymarked for this sport — `mtb`
#: for a rider, `hiking` or `running` for a runner. Other carried relations,
#: such as the cycle node networks, are connectors.
ROUTE_SIGNED = 0b01
ROUTE_OTHER = 0b10

#: Ways can belong to more routes than a `uint8` holds; beyond this the exact
#: count stops carrying information anyway.
MAX_ROUTE_COUNT = 255


@dataclass(frozen=True)
class EdgeFeatures:
    """Everything the cost model knows about each edge."""

    tech: npt.NDArray[np.float32]
    tech_up: npt.NDArray[np.float32]
    tech_conf: npt.NDArray[np.uint8]
    singletrack: npt.NDArray[np.float32]
    rollability: npt.NDArray[np.float32]
    flow: npt.NDArray[np.float32]
    traffic: npt.NDArray[np.float32]
    surface_class: npt.NDArray[np.uint8]
    way_class: npt.NDArray[np.uint8]
    n_signed_routes: npt.NDArray[np.uint8]
    route_flags: npt.NDArray[np.uint8]

    def __len__(self) -> int:
        """Number of edges described."""
        return int(self.tech.shape[0])

    @property
    def signed(self) -> npt.NDArray[np.bool_]:
        """Which edges lie on a route signed for this sport."""
        return (self.route_flags & ROUTE_SIGNED).astype(np.bool_)

    @property
    def connector(self) -> npt.NDArray[np.bool_]:
        """Which edges lie on some other carried route."""
        return (self.route_flags & ROUTE_OTHER).astype(np.bool_)

    def select(self, index: npt.NDArray[np.intp]) -> EdgeFeatures:
        """Return the features of a subset of edges, in the given order."""
        return EdgeFeatures(
            tech=self.tech[index],
            tech_up=self.tech_up[index],
            tech_conf=self.tech_conf[index],
            singletrack=self.singletrack[index],
            rollability=self.rollability[index],
            flow=self.flow[index],
            traffic=self.traffic[index],
            surface_class=self.surface_class[index],
            way_class=self.way_class[index],
            n_signed_routes=self.n_signed_routes[index],
            route_flags=self.route_flags[index],
        )

    def with_difficulty(
        self, tech: npt.NDArray[np.float32], tech_conf: npt.NDArray[np.uint8]
    ) -> EdgeFeatures:
        """Return a copy carrying revised technicality."""
        return replace(self, tech=tech, tech_conf=tech_conf)

    def confidence_at_least(self, level: TechConfidence) -> npt.NDArray[np.bool_]:
        """Which edges are known at least this well."""
        return self.tech_conf >= int(level)


def compute_features(
    segments: Sequence[Segment],
    route_relations: dict[int, RouteRelation],
    sport: Sport | None = None,
) -> EdgeFeatures:
    """Derive the feature vector for every segment."""
    sport = sport or active_sport()
    count = len(segments)
    signed_ids = {
        rid
        for rid, relation in route_relations.items()
        if relation.route_type in sport.signed_route_types
    }

    tech = np.zeros(count, dtype=np.float32)
    tech_up = np.zeros(count, dtype=np.float32)
    tech_conf = np.zeros(count, dtype=np.uint8)
    singletrack = np.zeros(count, dtype=np.float32)
    roll = np.zeros(count, dtype=np.float32)
    flow = np.zeros(count, dtype=np.float32)
    traffic = np.zeros(count, dtype=np.float32)
    surface = np.zeros(count, dtype=np.uint8)
    way = np.zeros(count, dtype=np.uint8)
    route_count = np.zeros(count, dtype=np.uint8)
    route_flags = np.zeros(count, dtype=np.uint8)

    for index, segment in enumerate(segments):
        tags = segment.tags
        surface_class = classify_surface(tags)
        difficulty = sport.difficulty(tags, surface_class)

        tech[index] = difficulty.tech
        tech_up[index] = difficulty.tech_up
        tech_conf[index] = int(difficulty.confidence)
        flow[index] = difficulty.flow
        surface[index] = int(surface_class)
        way[index] = int(classify_way(tags, surface_class))
        singletrack[index] = singletrack_score(tags, surface_class)
        roll[index] = rollability(tags, surface_class)
        traffic[index] = traffic_exposure(tags)

        if segment.routes:
            signed_here = segment.routes & signed_ids
            route_count[index] = min(len(signed_here), MAX_ROUTE_COUNT)
            flags = ROUTE_SIGNED if signed_here else 0
            if segment.routes - signed_here:
                flags |= ROUTE_OTHER
            route_flags[index] = flags

    return EdgeFeatures(
        tech=tech,
        tech_up=tech_up,
        tech_conf=tech_conf,
        singletrack=singletrack,
        rollability=roll,
        flow=flow,
        traffic=traffic,
        surface_class=surface,
        way_class=way,
        n_signed_routes=route_count,
        route_flags=route_flags,
    )
