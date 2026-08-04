"""Fetching Geofabrik extracts.

Downloads stream to a `.part` file and are only moved into place once the
published MD5 matches. A half-written extract that looks complete is worse
than no extract: the build would succeed and produce a graph with a hole in
it, which nothing downstream can detect.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx

from mtb_planner.config.logging import get_logger
from mtb_planner.core.errors import ExtractDownloadError
from mtb_planner.ingest.catalog import Region

logger = get_logger(__name__)

_CHUNK_BYTES = 1 << 20
_TIMEOUT_S = 60.0

#: How much of a file to read when checking it really is an extract. The type
#: string sits in the first blob header, well inside this.
_SNIFF_BYTES = 65536

#: Smallest plausible extract. A `.osm.pbf` carries a header blob before any
#: data, so anything this small is a failed or truncated download. Geofabrik
#: does not publish a checksum for every region, and without one an empty file
#: previously passed as "verified" and was then handed to the parser.
_MIN_PLAUSIBLE_BYTES = 1024

ProgressCallback = Callable[[int, int | None], None]


def _published_md5(client: httpx.Client, region: Region) -> str | None:
    """Fetch the published checksum, or `None` if the sidecar is unavailable."""
    try:
        response = client.get(region.md5_url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("No checksum available for %s: %s", region.slug, exc)
        return None
    # The sidecar is "<md5>  <filename>".
    return response.text.split(maxsplit=1)[0].strip().lower()


def _looks_like_pbf(path: Path) -> bool:
    """Whether a file is actually an OSM PBF.

    Geofabrik answers an unknown region path with its download index page and a
    200, so `raise_for_status` sees nothing wrong and an HTML document lands on
    disk named `.pbf`. It is only caught later, as a confusing parse error, or
    not at all — a silently missing extract just means a silently missing
    country.

    Every PBF carries the type string `OSMHeader` in its first blob header.
    """
    with path.open("rb") as handle:
        head = handle.read(_SNIFF_BYTES)
    if head.lstrip()[:1] in (b"<",):
        return False
    return b"OSMHeader" in head


def _file_md5(path: Path) -> str:
    """Compute a file's MD5 in bounded memory.

    MD5 because that is what Geofabrik publishes. This verifies that a
    download arrived intact, not that it came from a trusted party, hence
    `usedforsecurity=False`.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download_extract(
    region: Region,
    dest_dir: Path,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download one region's extract, verify it, and return its path.

    An existing file whose checksum already matches is left alone, so
    re-running the command is cheap.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / region.filename
    partial = target.with_suffix(target.suffix + ".part")

    with httpx.Client(timeout=_TIMEOUT_S, follow_redirects=True) as client:
        expected = _published_md5(client, region)

        if target.exists() and not force:
            size = target.stat().st_size
            if size < _MIN_PLAUSIBLE_BYTES or not _looks_like_pbf(target):
                logger.warning(
                    "%s is not a usable extract (%d bytes); downloading again",
                    target.name,
                    size,
                )
            elif expected is None:
                # Say what was actually checked. Claiming verification that did
                # not happen is how a zero-byte file reaches the parser.
                logger.info(
                    "%s already present (%.1f MB); no checksum published to verify it",
                    target.name,
                    size / 1e6,
                )
                return target
            elif _file_md5(target) == expected:
                logger.info("%s already present and verified", target.name)
                return target
            else:
                logger.warning("%s failed verification; re-downloading", target.name)

        try:
            with client.stream("GET", region.url) as response:
                response.raise_for_status()
                total = (
                    int(response.headers["content-length"])
                    if "content-length" in response.headers
                    else None
                )
                downloaded = 0
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes(_CHUNK_BYTES):
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if progress is not None:
                            progress(downloaded, total)
        except httpx.HTTPError as exc:
            partial.unlink(missing_ok=True)
            msg = f"Failed to download {region.url}: {exc}"
            raise ExtractDownloadError(msg) from exc

    if expected is not None:
        actual = _file_md5(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            msg = f"Checksum mismatch for {region.filename}: expected {expected}, got {actual}"
            raise ExtractDownloadError(msg)

    size = partial.stat().st_size
    if size < _MIN_PLAUSIBLE_BYTES or not _looks_like_pbf(partial):
        partial.unlink(missing_ok=True)
        msg = (
            f"{region.url} did not return an OSM extract ({size} bytes). "
            f"Check that {region.slug!r} is a real Geofabrik region: an unknown "
            f"path is answered with an HTML page rather than an error."
        )
        raise ExtractDownloadError(msg)

    partial.replace(target)
    logger.info("Downloaded %s (%.1f MB)", target.name, size / 1e6)
    return target
