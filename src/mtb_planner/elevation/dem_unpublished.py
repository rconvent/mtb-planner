"""Remembering which GLO-30 squares were never published.

GLO-30 covers land. A rectangular area of interest over the Channel or the
North Sea asks for tiles that do not exist, so `mtb fetch dem` treats a 404
as the shape of the coastline rather than a failure.

The build then has to tell those two cases apart. A tile nobody can download
is a fact about the archive; a tile that simply was not fetched is an
operator mistake that would leave a whole region flat. Both look identical on
disk — an absent file — so the fetch writes down what the archive refused,
and the build trusts that note instead of guessing.

Kept as a file beside the tiles rather than a table in the code: the set is a
property of the archive, not of this release, and nothing here should need a
network call to start a build.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from mtb_planner.config.logging import get_logger

logger = get_logger(__name__)

#: Written beside the tiles, so moving a DEM directory carries its own note.
UNPUBLISHED_FILENAME = "unpublished.json"

#: Bumped only if the file's shape changes. An unknown version is ignored
#: rather than guessed at, which reverts to demanding every tile.
SCHEMA_VERSION = 1


def registry_path(tile_dir: Path) -> Path:
    """Where the note lives for a given tile directory."""
    return tile_dir / UNPUBLISHED_FILENAME


def load_unpublished(tile_dir: Path) -> frozenset[str]:
    """Tile names the archive has no file for.

    A missing, malformed, or future-versioned file reads as empty. That is
    the safe direction: the build goes back to insisting on every tile, which
    fails loudly, rather than excusing a gap it cannot account for.
    """
    path = registry_path(tile_dir)
    if not path.exists():
        return frozenset()

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        version = int(document["schema"])
        names = document["unpublished"]
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Ignoring unreadable %s; every tile will be required", path)
        return frozenset()

    if version != SCHEMA_VERSION:
        logger.warning(
            "Ignoring %s written by schema %s; this build reads schema %s",
            path,
            version,
            SCHEMA_VERSION,
        )
        return frozenset()

    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        logger.warning("Ignoring malformed tile list in %s", path)
        return frozenset()

    return frozenset(names)


def record_unpublished(tile_dir: Path, names: Iterable[str]) -> frozenset[str]:
    """Add `names` to the note and return everything it now holds.

    Merged rather than replaced: a fetch covering one area must not forget
    what an earlier fetch learned about another.
    """
    merged = load_unpublished(tile_dir) | frozenset(names)
    if not merged:
        return merged

    document = {"schema": SCHEMA_VERSION, "unpublished": sorted(merged)}
    path = registry_path(tile_dir)
    partial = path.with_suffix(path.suffix + ".part")
    tile_dir.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)
    return merged
