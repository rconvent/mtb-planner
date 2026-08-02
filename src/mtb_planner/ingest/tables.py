"""Columnar storage for ingested OSM ways and nodes.

Dictionaries of per-way objects are what stop a country-scale build from
fitting in memory. At roughly three million rideable ways, a
`dict[int, RawWay]` holding tuples of Python integers costs well over a
gigabyte before any graph exists; the same data in flat arrays costs a few
hundred megabytes.

Node references use the same compressed-sparse-row shape as the finished
graph: one long `int64` array of refs plus per-way offsets into it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mtb_planner.core.units import COORD_SCALE
from mtb_planner.ingest.tags import WayTags
from mtb_planner.ingest.way_filter import BikeAccess, Direction


@dataclass(frozen=True, slots=True)
class WayRow:
    """A single way, materialised for convenient access."""

    way_id: int
    node_refs: npt.NDArray[np.int64]
    tags: WayTags
    access: BikeAccess
    direction: Direction
    oneway_corroborated: bool


@dataclass(frozen=True)
class NodeTable:
    """Every node referenced by a retained way.

    `ids` is sorted and unique, so lookups are a binary search and the other
    arrays are positional against it.
    """

    ids: npt.NDArray[np.int64]
    lat_e7: npt.NDArray[np.int32]
    lon_e7: npt.NDArray[np.int32]
    ref_count: npt.NDArray[np.int32]
    barrier: npt.NDArray[np.bool_]

    def __len__(self) -> int:
        """Number of nodes."""
        return int(self.ids.shape[0])

    def index_of(self, node_ids: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Positions of `node_ids`, vectorised.

        Callers must only pass ids known to be present; `contains` exists for
        the uncertain case. An absent id would silently land on a neighbour.
        """
        return np.searchsorted(self.ids, node_ids).astype(np.int64)

    def contains(self, node_ids: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
        """Whether each id is present, vectorised."""
        if len(self) == 0:
            return np.zeros(node_ids.shape, dtype=np.bool_)
        positions = np.searchsorted(self.ids, node_ids)
        clipped = np.clip(positions, 0, len(self) - 1)
        present: npt.NDArray[np.bool_] = (positions < len(self)) & (self.ids[clipped] == node_ids)
        return present

    def _position_of(self, node_id: int) -> int:
        """Row of a single node id."""
        position = int(np.searchsorted(self.ids, node_id))
        if position >= len(self) or int(self.ids[position]) != node_id:
            msg = f"Node {node_id} is not in the table"
            raise KeyError(msg)
        return position

    def __contains__(self, node_id: int) -> bool:
        """Whether a single node is present."""
        position = int(np.searchsorted(self.ids, node_id))
        return position < len(self) and int(self.ids[position]) == node_id

    def coord_of(self, node_id: int) -> tuple[int, int]:
        """Scaled `(lat_e7, lon_e7)` of a single node."""
        position = self._position_of(node_id)
        return int(self.lat_e7[position]), int(self.lon_e7[position])

    def ref_count_of(self, node_id: int) -> int:
        """How many retained ways reference a single node."""
        return int(self.ref_count[self._position_of(node_id)])

    def is_barrier(self, node_id: int) -> bool:
        """Whether a single node obstructs or regulates passage."""
        return bool(self.barrier[self._position_of(node_id)])

    def degrees(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Coordinates as WGS84 degrees, positional against `ids`."""
        return self.lat_e7 / COORD_SCALE, self.lon_e7 / COORD_SCALE


@dataclass(frozen=True)
class WayTable:
    """Every retained bike-relevant way.

    `ids` is sorted and unique. `node_offset` has one more entry than `ids`,
    so way `i` owns `node_refs[node_offset[i]:node_offset[i + 1]]`.
    """

    ids: npt.NDArray[np.int64]
    node_offset: npt.NDArray[np.int64]
    node_refs: npt.NDArray[np.int64]
    tags: tuple[WayTags, ...]
    access: npt.NDArray[np.uint8]
    direction: npt.NDArray[np.uint8]
    oneway_corroborated: npt.NDArray[np.bool_]

    def __len__(self) -> int:
        """Number of ways."""
        return int(self.ids.shape[0])

    def __contains__(self, way_id: int) -> bool:
        """Whether a way survived ingest."""
        position = int(np.searchsorted(self.ids, way_id))
        return position < len(self) and int(self.ids[position]) == way_id

    def index_of(self, way_id: int) -> int:
        """Row of a way id."""
        position = int(np.searchsorted(self.ids, way_id))
        if position >= len(self) or int(self.ids[position]) != way_id:
            msg = f"Way {way_id} is not in the table"
            raise KeyError(msg)
        return position

    def refs_at(self, index: int) -> npt.NDArray[np.int64]:
        """Node references of the way at `index`, as a view."""
        return self.node_refs[self.node_offset[index] : self.node_offset[index + 1]]

    def row(self, way_id: int) -> WayRow:
        """Materialise a single way by its OSM id."""
        index = self.index_of(way_id)
        return WayRow(
            way_id=way_id,
            node_refs=self.refs_at(index),
            tags=self.tags[index],
            access=BikeAccess(int(self.access[index])),
            direction=Direction(int(self.direction[index])),
            oneway_corroborated=bool(self.oneway_corroborated[index]),
        )

    @property
    def total_refs(self) -> int:
        """Total number of node references across all ways."""
        return int(self.node_refs.shape[0])
