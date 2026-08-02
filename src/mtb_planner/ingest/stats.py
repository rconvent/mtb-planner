"""Tag coverage report.

This exists because the scoring model rests on an assumption that has to be
measured, not guessed: how much of the network actually carries an explicit
`mtb:scale`. In Belgium the answer is roughly 3.4% of trail ways, which makes
inference the primary engine rather than a fallback.

Deliberately a separate streaming pass that accumulates counters only. It
never holds ways or nodes, so it runs over a full country extract in bounded
memory while a graph build over the same file would not.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import osmium
from osmium.osm import osm_entity_bits as EntityBits

from mtb_planner.config.logging import get_logger
from mtb_planner.core.errors import PbfParseError
from mtb_planner.ingest.tags import RETAINED_TAG_KEYS, WayTags, parse_mtb_scale
from mtb_planner.ingest.way_filter import is_bike_relevant

logger = get_logger(__name__)

#: Reported individually because each is load-bearing for the difficulty
#: model, and their coverage decides how much of it is inference.
HEADLINE_TAG_KEYS: tuple[str, ...] = (
    "mtb:scale",
    "mtb:scale:uphill",
    "mtb:scale:imba",
    "mtb:type",
    "sac_scale",
    "tracktype",
    "surface",
    "smoothness",
    "width",
    "trail_visibility",
    "informal",
    "incline",
    "oneway",
    "oneway:bicycle",
)

#: Ways whose difficulty the model must estimate when it is not tagged.
TRAIL_HIGHWAYS: frozenset[str] = frozenset({"path", "track", "bridleway"})


@dataclass
class TagStats:
    """Counts over one or more extracts.

    Mutable by design. This is an accumulator updated once per way, and at
    country scale that is tens of millions of updates; rebuilding a frozen
    dataclass holding ten counters that often would dominate the runtime.
    Callers should treat an instance as read-only once `collect_tag_stats`
    has returned it.
    """

    files: tuple[str, ...]
    total_ways: int = 0
    bike_relevant_ways: int = 0
    trail_ways: int = 0
    highway_counts: Counter[str] = field(default_factory=Counter)
    tag_counts: Counter[str] = field(default_factory=Counter)
    trail_tag_counts: Counter[str] = field(default_factory=Counter)
    mtb_scale_values: Counter[str] = field(default_factory=Counter)
    unparseable_mtb_scale: Counter[str] = field(default_factory=Counter)
    bicycle_values: Counter[str] = field(default_factory=Counter)
    access_values: Counter[str] = field(default_factory=Counter)
    route_relations: Counter[str] = field(default_factory=Counter)

    @property
    def mtb_scale_coverage(self) -> float:
        """Fraction of trail ways carrying an explicit `mtb:scale`.

        The single most important number in the project: the share of the
        network whose difficulty is surveyed rather than inferred.
        """
        if self.trail_ways == 0:
            return 0.0
        return self.trail_tag_counts["mtb:scale"] / self.trail_ways

    def to_dict(self) -> dict[str, Any]:
        """Serialise for `--json` output."""
        return {
            "files": list(self.files),
            "total_ways": self.total_ways,
            "bike_relevant_ways": self.bike_relevant_ways,
            "trail_ways": self.trail_ways,
            "mtb_scale_coverage": round(self.mtb_scale_coverage, 5),
            "highway_counts": dict(self.highway_counts.most_common()),
            "tag_counts": dict(self.tag_counts.most_common()),
            "trail_tag_counts": dict(self.trail_tag_counts.most_common()),
            "mtb_scale_values": dict(self.mtb_scale_values.most_common()),
            "unparseable_mtb_scale": dict(self.unparseable_mtb_scale.most_common()),
            "bicycle_values": dict(self.bicycle_values.most_common()),
            "access_values": dict(self.access_values.most_common()),
            "route_relations": dict(self.route_relations.most_common()),
        }


def _resolved_paths(paths: Iterable[Path | str]) -> list[Path]:
    resolved = [Path(p) for p in paths]
    if not resolved:
        msg = "No extract paths given"
        raise PbfParseError(msg)
    for path in resolved:
        if not path.exists():
            msg = f"Extract not found: {path}"
            raise PbfParseError(msg)
    return resolved


def _count_ways(path: Path, stats: TagStats) -> None:
    """Accumulate way-level counters for a single extract."""
    for way in osmium.FileProcessor(str(path), EntityBits.WAY):
        stats.total_ways += 1
        raw = dict(way.tags)

        highway = raw.get("highway")
        if highway is not None:
            stats.highway_counts[highway] += 1

        for key in RETAINED_TAG_KEYS:
            if key in raw:
                stats.tag_counts[key] += 1

        if (bicycle := raw.get("bicycle")) is not None:
            stats.bicycle_values[bicycle] += 1
        if (access := raw.get("access")) is not None:
            stats.access_values[access] += 1

        if (scale := raw.get("mtb:scale")) is not None:
            stats.mtb_scale_values[scale] += 1
            if parse_mtb_scale(scale) is None:
                stats.unparseable_mtb_scale[scale] += 1

        if is_bike_relevant(WayTags.from_osm(raw)):
            stats.bike_relevant_ways += 1

        if highway in TRAIL_HIGHWAYS:
            stats.trail_ways += 1
            for key in HEADLINE_TAG_KEYS:
                if key in raw:
                    stats.trail_tag_counts[key] += 1


def _count_relations(path: Path, stats: TagStats) -> None:
    """Accumulate route-relation counters for a single extract."""
    for relation in osmium.FileProcessor(str(path), EntityBits.RELATION):
        tags = relation.tags
        if tags.get("type") != "route":
            continue
        if (route := tags.get("route")) is not None:
            stats.route_relations[route] += 1


def collect_tag_stats(paths: Iterable[Path | str]) -> TagStats:
    """Stream one or more extracts and count tag usage.

    Way counts include every way in the file, matching how taginfo reports
    them, so the output can be compared against published figures directly.
    """
    resolved = _resolved_paths(paths)

    stats = TagStats(files=tuple(p.name for p in resolved))
    for path in resolved:
        _count_ways(path, stats)
        _count_relations(path, stats)

    logger.info(
        "Scanned %d ways (%d bike-relevant, %d trails); mtb:scale on %.2f%% of trails",
        stats.total_ways,
        stats.bike_relevant_ways,
        stats.trail_ways,
        stats.mtb_scale_coverage * 100,
    )
    return stats


def format_tag_stats(stats: TagStats, top_n: int = 12) -> Sequence[tuple[str, str]]:
    """Render the headline findings as label/value pairs for display."""
    coverage = stats.mtb_scale_coverage
    rows: list[tuple[str, str]] = [
        ("extracts", ", ".join(stats.files)),
        ("ways scanned", f"{stats.total_ways:,}"),
        ("bike-relevant ways", f"{stats.bike_relevant_ways:,}"),
        ("trail ways (path/track/bridleway)", f"{stats.trail_ways:,}"),
        (
            "mtb:scale on trails",
            f"{stats.trail_tag_counts['mtb:scale']:,}  ({coverage * 100:.2f}%)",
        ),
        ("route=mtb relations", f"{stats.route_relations['mtb']:,}"),
        ("route=bicycle relations", f"{stats.route_relations['bicycle']:,}"),
    ]
    rows.extend(
        (f"highway={name}", f"{count:,}") for name, count in stats.highway_counts.most_common(top_n)
    )
    rows.extend(
        (f"tag {key}", f"{stats.tag_counts[key]:,}")
        for key in HEADLINE_TAG_KEYS
        if stats.tag_counts[key]
    )
    if stats.unparseable_mtb_scale:
        rows.append(
            (
                "unparseable mtb:scale",
                ", ".join(f"{v!r}x{c}" for v, c in stats.unparseable_mtb_scale.most_common(5)),
            )
        )
    return rows
