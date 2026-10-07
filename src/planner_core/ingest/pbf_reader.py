"""Three-pass streaming reader for `.osm.pbf` extracts.

Passes are explicit rather than using pyosmium's `with_locations()`, because
that helper's node cache is per-file and would break the moment a second
extract is added.

Two things keep this within memory at country scale:

* An `osmium.IdTracker` records which nodes the retained ways reference, and
  its `id_filter()` is applied inside libosmium. Only referenced nodes ever
  cross into Python — on Luxembourg that is 0.9M of 4.2M, and it is four
  times faster than filtering on the Python side.
* Everything accumulates into `array.array` buffers and lands in columnar
  NumPy tables. A `dict[int, RawWay]` of three million ways costs over a
  gigabyte; the same data in flat arrays costs a few hundred megabytes.

Multiple extracts merge here, not via `osmium-tool` — that binary is only
distributed for Windows through conda, and the merge rules are specific
enough to be worth owning:

* Ways are deduplicated by OSM id, keeping the copy with the most node
  references. Border-clipped copies are truncated, so the longest one wins.
* Relation members are unioned across copies; a route clipped at a border
  loses members in one extract and has them in the other.
* A way is only dropped by the bounding box when *every* node lies outside.
  Dropping on the first out-of-box node would build routing walls precisely
  along the Ardennes borders, where the good riding is.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import osmium
from osmium.osm import Node, Relation, Way
from osmium.osm import osm_entity_bits as EntityBits

from planner_core.config.logging import get_logger
from planner_core.core.arrays import ragged_indices
from planner_core.core.errors import PbfParseError
from planner_core.core.geo import BBox
from planner_core.core.units import COORD_SCALE
from planner_core.ingest.tables import NodeTable, WayTable
from planner_core.ingest.tags import WayTags
from planner_core.sport import Access, Sport, active_sport

logger = get_logger(__name__)

#: Node tags that force a junction split, because they cost time to pass or
#: change what is legal. Barriers also anchor turn penalties later.
_BARRIER_HIGHWAYS: frozenset[str] = frozenset(
    {"traffic_signals", "crossing", "stop", "give_way", "mini_roundabout"}
)

#: `barrier` values that do not actually obstruct anything.
_NON_BARRIERS: frozenset[str] = frozenset({"no", "entrance"})

#: A way needs at least two nodes to have any length.
_MIN_WAY_NODES = 2


@dataclass(frozen=True, slots=True)
class RouteRelation:
    """A waymarked route: a mountain-bike trail, a cycle network, a hiking path.

    Which types are read is the sport's choice. A closed signed relation is a
    loop somebody physically marked on the ground, which makes it a better seed
    than anything an algorithm invents.
    """

    relation_id: int
    route_type: str
    name: str | None
    ref: str | None
    network: str | None
    operator: str | None
    colour: str | None
    way_ids: frozenset[int]


@dataclass(frozen=True)
class IngestResult:
    """Everything read from the extracts, before any graph is built."""

    nodes: NodeTable
    ways: WayTable
    route_relations: dict[int, RouteRelation]
    way_to_routes: dict[int, frozenset[int]]
    dropped_incomplete_ways: int = 0
    dropped_outside_bbox_ways: int = 0

    @property
    def n_ways(self) -> int:
        """Number of retained ways."""
        return len(self.ways)

    @property
    def n_nodes(self) -> int:
        """Number of nodes with known coordinates."""
        return len(self.nodes)

    def signed_routes(self, sport: Sport | None = None) -> dict[int, RouteRelation]:
        """Only the relations signed for this sport."""
        signed = (sport or active_sport()).signed_route_types
        return {rid: rel for rid, rel in self.route_relations.items() if rel.route_type in signed}


def _as_paths(paths: Iterable[Path | str]) -> list[Path]:
    """Normalise inputs and fail early on anything missing."""
    resolved = [Path(p) for p in paths]
    if not resolved:
        msg = "No extract paths given"
        raise PbfParseError(msg)
    for path in resolved:
        if not path.exists():
            msg = f"Extract not found: {path}"
            raise PbfParseError(msg)
    return resolved


# --------------------------------------------------------------- pass R


def _route_type_of(tags: Mapping[str, str], carried: frozenset[str]) -> str | None:
    """Return the relation's route type if it is one we carry."""
    if tags.get("type") != "route":
        return None
    route_type = (tags.get("route") or "").split(";", 1)[0].strip().lower()
    return route_type if route_type in carried else None


def read_route_relations(
    paths: Sequence[Path], sport: Sport
) -> tuple[dict[int, RouteRelation], dict[int, frozenset[int]]]:
    """Pass R — collect the route relations this sport carries.

    Members are unioned across extracts so a border-clipped relation is made
    whole again rather than silently losing half its ways.
    """
    members: dict[int, set[int]] = {}
    meta: dict[int, tuple[str, dict[str, str]]] = {}

    for path in paths:
        for entity in osmium.FileProcessor(str(path), EntityBits.RELATION):
            # The entity filter guarantees the type; the iterator cannot say so.
            relation = cast("Relation", entity)
            tags = dict(relation.tags)
            route_type = _route_type_of(tags, sport.route_types)
            if route_type is None:
                continue

            rid = relation.id
            meta.setdefault(rid, (route_type, tags))
            members.setdefault(rid, set()).update(
                member.ref for member in relation.members if member.type == "w"
            )

    relations: dict[int, RouteRelation] = {}
    way_to_routes: dict[int, set[int]] = {}
    for rid, (route_type, tags) in meta.items():
        way_ids = frozenset(members.get(rid, ()))
        relations[rid] = RouteRelation(
            relation_id=rid,
            route_type=route_type,
            name=tags.get("name"),
            ref=tags.get("ref"),
            network=tags.get("network"),
            operator=tags.get("operator"),
            colour=tags.get("colour") or tags.get("color"),
            way_ids=way_ids,
        )
        for way_id in way_ids:
            way_to_routes.setdefault(way_id, set()).add(rid)

    logger.info(
        "Pass R: %d route relations (%d signed) across %d extract(s)",
        len(relations),
        sum(1 for r in relations.values() if r.route_type in sport.signed_route_types),
        len(paths),
    )
    return relations, {wid: frozenset(rids) for wid, rids in way_to_routes.items()}


# --------------------------------------------------------------- pass W


class _WayAccumulator:
    """Collects retained ways into compact buffers, deduplicating by id."""

    def __init__(self) -> None:
        self.ids = array("q")
        self.offsets = array("q", [0])
        self.refs = array("q")
        self.access = array("B")
        self.direction = array("B")
        self.corroborated = array("B")
        self.tags: list[WayTags] = []
        # Row currently winning for each way id, so a border-clipped duplicate
        # never displaces the complete copy.
        self._best: dict[int, int] = {}
        # Most ways share a handful of tag combinations; interning turns
        # millions of WayTags objects into thousands.
        self._interned: dict[WayTags, WayTags] = {}

    def intern(self, tags: WayTags) -> WayTags:
        """Return a shared instance of an equal `WayTags`."""
        return self._interned.setdefault(tags, tags)

    def length_of(self, row: int) -> int:
        """Number of node references held by a row."""
        return self.offsets[row + 1] - self.offsets[row]

    def add(
        self,
        way_id: int,
        node_refs: Sequence[int],
        tags: WayTags,
        access: int,
        direction: int,
        corroborated: bool,
    ) -> bool:
        """Record a way, unless a longer copy of it is already held."""
        existing = self._best.get(way_id)
        if existing is not None and self.length_of(existing) >= len(node_refs):
            return False

        row = len(self.ids)
        self.ids.append(way_id)
        self.refs.extend(node_refs)
        self.offsets.append(len(self.refs))
        self.access.append(access)
        self.direction.append(direction)
        self.corroborated.append(1 if corroborated else 0)
        self.tags.append(tags)
        self._best[way_id] = row
        return True

    def finalise(self) -> WayTable:
        """Compact the winning rows into a sorted `WayTable`."""
        if not self._best:
            return WayTable(
                ids=np.empty(0, dtype=np.int64),
                node_offset=np.zeros(1, dtype=np.int64),
                node_refs=np.empty(0, dtype=np.int64),
                tags=(),
                access=np.empty(0, dtype=np.uint8),
                direction=np.empty(0, dtype=np.uint8),
                oneway_corroborated=np.empty(0, dtype=np.bool_),
            )

        all_ids = np.frombuffer(self.ids, dtype=np.int64)
        all_offsets = np.frombuffer(self.offsets, dtype=np.int64)
        all_refs = np.frombuffer(self.refs, dtype=np.int64)

        rows = np.fromiter(self._best.values(), dtype=np.int64, count=len(self._best))
        rows = rows[np.argsort(all_ids[rows], kind="stable")]

        starts = all_offsets[rows]
        lengths = all_offsets[rows + 1] - starts

        return WayTable(
            ids=all_ids[rows].copy(),
            node_offset=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            node_refs=all_refs[ragged_indices(starts, lengths)].copy(),
            tags=tuple(self.tags[int(row)] for row in rows),
            access=np.frombuffer(self.access, dtype=np.uint8)[rows].copy(),
            direction=np.frombuffer(self.direction, dtype=np.uint8)[rows].copy(),
            oneway_corroborated=np.frombuffer(self.corroborated, dtype=np.uint8)[rows].astype(
                np.bool_
            ),
        )


def read_ways(paths: Sequence[Path], sport: Sport) -> tuple[WayTable, osmium.IdTracker]:
    """Pass W — collect the ways this sport may use, and track the nodes they need.

    The tracker is handed to pass N so libosmium can drop every unreferenced
    node before it reaches Python.
    """
    accumulator = _WayAccumulator()
    tracker = osmium.IdTracker()

    for path in paths:
        contributed = 0
        for entity in osmium.FileProcessor(str(path), EntityBits.WAY):
            way = cast("Way", entity)
            tags = WayTags.from_osm(dict(way.tags))
            access = sport.access(tags)
            if access is Access.NO:
                continue

            node_refs = [node.ref for node in way.nodes]
            if len(node_refs) < _MIN_WAY_NODES:
                continue

            rule = sport.direction(tags)
            if accumulator.add(
                way_id=way.id,
                node_refs=node_refs,
                tags=accumulator.intern(tags),
                access=int(access),
                direction=int(rule.direction),
                corroborated=rule.corroborated,
            ):
                tracker.add_references(way)
                contributed += 1
        logger.info("Pass W: %s contributed %d %s ways", path.name, contributed, sport.id)

    ways = accumulator.finalise()
    logger.info(
        "Pass W: %d ways, %d node references, %d distinct tag combinations",
        len(ways),
        ways.total_refs,
        len(accumulator._interned),
    )
    return ways, tracker


# --------------------------------------------------------------- pass N


def _is_barrier(barrier: str | None, highway: str | None) -> bool:
    """Whether a node obstructs or regulates passage.

    Takes the two values rather than a mapping so the caller need not
    materialise a dict for every node in the extract.
    """
    if barrier is not None and barrier.lower() not in _NON_BARRIERS:
        return True
    return highway in _BARRIER_HIGHWAYS


def read_nodes(paths: Sequence[Path], tracker: osmium.IdTracker) -> NodeTable:
    """Pass N — read coordinates for referenced nodes, and note barriers.

    Coordinates are stored scaled to 1e-7 degrees as integers, matching OSM's
    own encoding. `float32` would fit in the same four bytes but carries about
    0.4 m of error at Belgian latitudes, which quietly corrupts every length.
    """
    ids = array("q")
    lat_e7 = array("i")
    lon_e7 = array("i")
    barrier = array("B")

    for path in paths:
        processor = osmium.FileProcessor(str(path), EntityBits.NODE).with_filter(
            tracker.id_filter()
        )
        for entity in processor:
            node = cast("Node", entity)
            location = node.location
            ids.append(node.id)
            lat_e7.append(round(location.lat * COORD_SCALE))
            lon_e7.append(round(location.lon * COORD_SCALE))
            barrier.append(
                1 if _is_barrier(node.tags.get("barrier"), node.tags.get("highway")) else 0
            )

    raw_ids = np.frombuffer(ids, dtype=np.int64)
    # A node present in two overlapping extracts appears twice; the copies are
    # identical, so the first occurrence is as good as any.
    unique_ids, first = np.unique(raw_ids, return_index=True)

    table = NodeTable(
        ids=unique_ids,
        lat_e7=np.frombuffer(lat_e7, dtype=np.int32)[first].copy(),
        lon_e7=np.frombuffer(lon_e7, dtype=np.int32)[first].copy(),
        # Filled in once the surviving ways are known.
        ref_count=np.zeros(unique_ids.shape[0], dtype=np.int32),
        barrier=np.frombuffer(barrier, dtype=np.uint8)[first].astype(np.bool_),
    )
    logger.info(
        "Pass N: %d node coordinates, %d barrier nodes",
        len(table),
        int(table.barrier.sum()),
    )
    return table


# --------------------------------------------------------------- assembly


def _keep_mask(
    ways: WayTable, nodes: NodeTable, bbox: BBox | None
) -> tuple[npt.NDArray[np.bool_], int, int]:
    """Decide which ways survive coordinate completeness and the bounding box."""
    starts = ways.node_offset[:-1]
    lengths = np.diff(ways.node_offset)

    present = nodes.contains(ways.node_refs)
    complete = np.add.reduceat(present.astype(np.int64), starts) == lengths

    if bbox is None:
        return complete, int((~complete).sum()), 0

    positions = np.clip(nodes.index_of(ways.node_refs), 0, max(len(nodes) - 1, 0))
    lat = nodes.lat_e7[positions] / COORD_SCALE
    lon = nodes.lon_e7[positions] / COORD_SCALE
    inside = present & (
        (lat >= bbox.min_lat)
        & (lat <= bbox.max_lat)
        & (lon >= bbox.min_lon)
        & (lon <= bbox.max_lon)
    )
    # Only drop when the whole way is outside — a way with one node inside
    # still carries riders across the boundary.
    touches = np.add.reduceat(inside.astype(np.int64), starts) > 0

    keep = complete & touches
    return keep, int((~complete).sum()), int((complete & ~touches).sum())


def _select_ways(ways: WayTable, keep: npt.NDArray[np.bool_]) -> WayTable:
    """Return a table holding only the ways selected by `keep`."""
    if bool(keep.all()):
        return ways

    lengths = np.diff(ways.node_offset)[keep]
    return WayTable(
        ids=ways.ids[keep],
        node_offset=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        node_refs=ways.node_refs[np.repeat(keep, np.diff(ways.node_offset))],
        tags=tuple(tag for tag, wanted in zip(ways.tags, keep, strict=True) if wanted),
        access=ways.access[keep],
        direction=ways.direction[keep],
        oneway_corroborated=ways.oneway_corroborated[keep],
    )


def _with_reference_counts(nodes: NodeTable, ways: WayTable) -> NodeTable:
    """Count how many surviving ways reference each node.

    Counting before the drop would invent junctions on ways that are no
    longer in the graph.
    """
    counts = np.zeros(len(nodes), dtype=np.int32)
    if ways.total_refs:
        positions = nodes.index_of(ways.node_refs)
        np.add.at(counts, positions, 1)
    return NodeTable(
        ids=nodes.ids,
        lat_e7=nodes.lat_e7,
        lon_e7=nodes.lon_e7,
        ref_count=counts,
        barrier=nodes.barrier,
    )


def read_extracts(
    paths: Iterable[Path | str], bbox: BBox | None = None, sport: Sport | None = None
) -> IngestResult:
    """Read one or more `.osm.pbf` extracts into ways, nodes, and routes."""
    sport = sport or active_sport()
    resolved = _as_paths(paths)

    route_relations, way_to_routes = read_route_relations(resolved, sport)
    ways, tracker = read_ways(resolved, sport)
    if len(ways) == 0:
        msg = f"No {sport.id} ways found in {', '.join(p.name for p in resolved)}"
        raise PbfParseError(msg)

    nodes = read_nodes(resolved, tracker)

    keep, dropped_incomplete, dropped_outside = _keep_mask(ways, nodes, bbox)
    if dropped_incomplete:
        logger.warning(
            "Dropped %d way(s) referencing nodes absent from every extract", dropped_incomplete
        )
    if dropped_outside:
        logger.info("Dropped %d way(s) entirely outside the area of interest", dropped_outside)

    if not bool(keep.any()):
        msg = f"Every {sport.id} way was dropped; check the bounding box"
        raise PbfParseError(msg)

    kept = _select_ways(ways, keep)
    return IngestResult(
        nodes=_with_reference_counts(nodes, kept),
        ways=kept,
        route_relations=route_relations,
        way_to_routes=way_to_routes,
        dropped_incomplete_ways=dropped_incomplete,
        dropped_outside_bbox_ways=dropped_outside,
    )
