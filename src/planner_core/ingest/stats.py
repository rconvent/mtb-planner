"""Tag coverage report.

This exists because the scoring model rests on an assumption that has to be
measured, not guessed: how much of the network actually carries an explicit
difficulty grade — `mtb:scale` for a rider, `sac_scale` for a runner. In
Belgium the answer for `mtb:scale` is roughly 3.4% of trail ways, which makes
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

from planner_core.config.logging import get_logger
from planner_core.core.errors import PbfParseError
from planner_core.ingest.tags import RETAINED_TAG_KEYS, WayTags
from planner_core.sport import Sport, active_sport

logger = get_logger(__name__)

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
    #: The key whose explicit value counts as surveyed difficulty, and the
    #: mode-specific access key, both chosen by the sport.
    surveyed_tag: str = "mtb:scale"
    access_tag: str = "bicycle"
    headline_tags: tuple[str, ...] = ()
    total_ways: int = 0
    relevant_ways: int = 0
    trail_ways: int = 0
    highway_counts: Counter[str] = field(default_factory=Counter)
    tag_counts: Counter[str] = field(default_factory=Counter)
    trail_tag_counts: Counter[str] = field(default_factory=Counter)
    surveyed_values: Counter[str] = field(default_factory=Counter)
    unparseable_surveyed: Counter[str] = field(default_factory=Counter)
    mode_values: Counter[str] = field(default_factory=Counter)
    access_values: Counter[str] = field(default_factory=Counter)
    route_relations: Counter[str] = field(default_factory=Counter)

    @property
    def surveyed_coverage(self) -> float:
        """Fraction of trail ways carrying an explicit difficulty grade.

        The single most important number in the project: the share of the
        network whose difficulty is surveyed rather than inferred.
        """
        if self.trail_ways == 0:
            return 0.0
        return self.trail_tag_counts[self.surveyed_tag] / self.trail_ways

    def to_dict(self) -> dict[str, Any]:
        """Serialise for `--json` output."""
        return {
            "files": list(self.files),
            "total_ways": self.total_ways,
            "surveyed_tag": self.surveyed_tag,
            "relevant_ways": self.relevant_ways,
            "trail_ways": self.trail_ways,
            "surveyed_coverage": round(self.surveyed_coverage, 5),
            "highway_counts": dict(self.highway_counts.most_common()),
            "tag_counts": dict(self.tag_counts.most_common()),
            "trail_tag_counts": dict(self.trail_tag_counts.most_common()),
            "surveyed_values": dict(self.surveyed_values.most_common()),
            "unparseable_surveyed": dict(self.unparseable_surveyed.most_common()),
            "access_tag": self.access_tag,
            "mode_values": dict(self.mode_values.most_common()),
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


def _count_ways(path: Path, stats: TagStats, sport: Sport) -> None:
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

        if (mode := raw.get(stats.access_tag)) is not None:
            stats.mode_values[mode] += 1
        if (access := raw.get("access")) is not None:
            stats.access_values[access] += 1

        if (scale := raw.get(stats.surveyed_tag)) is not None:
            stats.surveyed_values[scale] += 1
            if sport.parse_surveyed(scale) is None:
                stats.unparseable_surveyed[scale] += 1

        if sport.is_relevant(WayTags.from_osm(raw)):
            stats.relevant_ways += 1

        if highway in TRAIL_HIGHWAYS:
            stats.trail_ways += 1
            for key in stats.headline_tags:
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


def collect_tag_stats(paths: Iterable[Path | str], sport: Sport | None = None) -> TagStats:
    """Stream one or more extracts and count tag usage.

    Way counts include every way in the file, matching how taginfo reports
    them, so the output can be compared against published figures directly.
    """
    sport = sport or active_sport()
    resolved = _resolved_paths(paths)

    stats = TagStats(
        files=tuple(p.name for p in resolved),
        surveyed_tag=sport.surveyed_tag,
        access_tag=sport.access_tag,
        headline_tags=sport.headline_tags,
    )
    for path in resolved:
        _count_ways(path, stats, sport)
        _count_relations(path, stats)

    logger.info(
        "Scanned %d ways (%d usable, %d trails); %s on %.2f%% of trails",
        stats.total_ways,
        stats.relevant_ways,
        stats.trail_ways,
        stats.surveyed_tag,
        stats.surveyed_coverage * 100,
    )
    return stats


def format_tag_stats(
    stats: TagStats, top_n: int = 12, sport: Sport | None = None
) -> Sequence[tuple[str, str]]:
    """Render the headline findings as label/value pairs for display."""
    sport = sport or active_sport()
    coverage = stats.surveyed_coverage
    rows: list[tuple[str, str]] = [
        ("extracts", ", ".join(stats.files)),
        ("ways scanned", f"{stats.total_ways:,}"),
        (f"{sport.id} ways", f"{stats.relevant_ways:,}"),
        ("trail ways (path/track/bridleway)", f"{stats.trail_ways:,}"),
        (
            f"{stats.surveyed_tag} on trails",
            f"{stats.trail_tag_counts[stats.surveyed_tag]:,}  ({coverage * 100:.2f}%)",
        ),
        *(
            (f"route={route} relations", f"{stats.route_relations[route]:,}")
            for route in sorted(sport.route_types)
        ),
    ]
    rows.extend(
        (f"highway={name}", f"{count:,}") for name, count in stats.highway_counts.most_common(top_n)
    )
    rows.extend(
        (f"tag {key}", f"{stats.tag_counts[key]:,}")
        for key in stats.headline_tags
        if stats.tag_counts[key]
    )
    if stats.unparseable_surveyed:
        rows.append(
            (
                f"unparseable {stats.surveyed_tag}",
                ", ".join(f"{v!r}x{c}" for v, c in stats.unparseable_surveyed.most_common(5)),
            )
        )
    return rows
