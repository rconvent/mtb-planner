"""Cutting ways into segments between junctions.

A node is a junction when two or more retained ways reference it, when it
terminates a way, or when it carries a barrier. Barriers split because they
cost time to pass and will later anchor turn penalties; leaving them buried
inside an edge would make that impossible to express.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from planner_core.config.logging import get_logger
from planner_core.ingest.tables import NodeTable, WayTable
from planner_core.ingest.tags import WayTags
from planner_core.sport import Direction

logger = get_logger(__name__)

#: Fewer nodes than this is not a traversable run.
_MIN_SEGMENT_NODES = 2


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of nodes between two junctions, carrying one way's properties."""

    node_ids: npt.NDArray[np.int64]
    way_id: int
    tags: WayTags
    access: int
    direction: int
    routes: frozenset[int]

    @property
    def u(self) -> int:
        """First node."""
        return int(self.node_ids[0])

    @property
    def v(self) -> int:
        """Last node."""
        return int(self.node_ids[-1])

    @property
    def is_loop(self) -> bool:
        """Whether the segment returns to where it started."""
        return self.u == self.v

    def reversed(self) -> Segment:
        """Return the same segment travelled the other way.

        Direction flips with the geometry: a forward-only descent becomes a
        backward-only one when its node order is reversed.
        """
        flipped = self.direction
        if self.direction == Direction.FORWARD:
            flipped = int(Direction.BACKWARD)
        elif self.direction == Direction.BACKWARD:
            flipped = int(Direction.FORWARD)
        return Segment(
            node_ids=self.node_ids[::-1],
            way_id=self.way_id,
            tags=self.tags,
            access=self.access,
            direction=flipped,
            routes=self.routes,
        )


def effective_direction(direction: int, corroborated: bool) -> int:
    """The direction actually enforced when building arcs.

    An uncorroborated `oneway` on a path is treated as bidirectional. A single
    mis-tagged way can otherwise sever a valley silently, and there is no cost
    model yet in which to express it as the heavy penalty it should become.
    """
    return direction if corroborated else int(Direction.BOTH)


def junction_mask(nodes: NodeTable, ways: WayTable) -> npt.NDArray[np.bool_]:
    """Mark which nodes ways must be cut at, positional against `nodes.ids`."""
    is_junction = (nodes.ref_count >= 2) | nodes.barrier

    if len(ways):
        starts = ways.node_offset[:-1]
        ends = ways.node_offset[1:] - 1
        endpoints = np.concatenate([ways.node_refs[starts], ways.node_refs[ends]])
        is_junction[nodes.index_of(endpoints)] = True

    return is_junction


def split_ways(
    nodes: NodeTable,
    ways: WayTable,
    way_to_routes: dict[int, frozenset[int]],
) -> list[Segment]:
    """Cut every way into segments at its junctions."""
    is_junction = junction_mask(nodes, ways)
    no_routes: frozenset[int] = frozenset()
    segments: list[Segment] = []

    for index in range(len(ways)):
        refs = ways.refs_at(index)
        cuts = np.flatnonzero(is_junction[nodes.index_of(refs)])
        if cuts.shape[0] < _MIN_SEGMENT_NODES:
            # Endpoints are always junctions, so this only happens on
            # malformed input; keeping the whole run loses no geometry.
            cuts = np.array([0, refs.shape[0] - 1], dtype=np.int64)

        way_id = int(ways.ids[index])
        tags = ways.tags[index]
        access = int(ways.access[index])
        direction = effective_direction(
            int(ways.direction[index]), bool(ways.oneway_corroborated[index])
        )
        routes = way_to_routes.get(way_id, no_routes)

        for start, stop in pairwise(cuts):
            segments.append(
                Segment(
                    node_ids=refs[start : stop + 1],
                    way_id=way_id,
                    tags=tags,
                    access=access,
                    direction=direction,
                    routes=routes,
                )
            )

    logger.info("Split %d ways into %d segments", len(ways), len(segments))
    return segments
