"""`mtb fetch` — download the source data a build needs."""

from __future__ import annotations

from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TransferSpeedColumn,
)

from mtb_planner.cli.parsing import parse_bbox
from mtb_planner.config.aoi import DEFAULT_CONFIG_PATH, load_aoi_config
from mtb_planner.config.logging import get_logger
from mtb_planner.config.settings import get_settings
from mtb_planner.core.errors import ExtractDownloadError, MtbPlannerError
from mtb_planner.elevation.dem_catalog import tiles_for_bbox
from mtb_planner.ingest.catalog import DEFAULT_AOI_REGIONS, Region
from mtb_planner.ingest.download import download_extract

app = typer.Typer(help="Download OpenStreetMap extracts and elevation tiles.")
console = Console()
logger = get_logger(__name__)

#: A GLO-30 tile is around 42 MB, so allow generously for a slow link.
_TILE_TIMEOUT_S = 300.0
_CHUNK_BYTES = 1 << 20


@app.command("osm")
def fetch_osm(
    regions: list[str] = typer.Argument(
        None,
        help="Geofabrik region slugs, e.g. europe/luxembourg. Defaults to the whole AOI.",
    ),
    dest: Path | None = typer.Option(
        None, "--dest", help="Where to write. Defaults to MTB_DATA_DIR/osm."
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if already verified."),
) -> None:
    """Download one or more Geofabrik extracts.

    Each file is verified against its published MD5 before being moved into
    place, so an interrupted download can never be mistaken for a complete one.
    """
    settings = get_settings()
    target_dir = dest or settings.osm_dir
    slugs = regions or list(DEFAULT_AOI_REGIONS)

    console.print(f"Fetching {len(slugs)} extract(s) into [bold]{target_dir}[/bold]")

    for slug in slugs:
        try:
            region = Region(slug)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(region.name, total=None)

            def advance(done: int, total: int | None, task_id: TaskID = task) -> None:
                progress.update(task_id, completed=done, total=total)

            try:
                path = download_extract(region, target_dir, force=force, progress=advance)
            except MtbPlannerError as exc:
                console.print(f"[red]{slug}: {exc}[/red]")
                raise typer.Exit(code=1) from exc

        console.print(f"  [green]OK[/green] {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


def _download_tile(url: str, destination: Path) -> None:
    """Stream one tile to disk, leaving nothing behind on failure."""
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with httpx.stream("GET", url, timeout=_TILE_TIMEOUT_S, follow_redirects=True) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        msg = f"Failed to download {url}: {exc}"
        raise ExtractDownloadError(msg) from exc
    partial.replace(destination)


@app.command("dem")
def fetch_dem(
    config: Path = typer.Option(
        None, "--config", help=f"Area-of-interest config. Defaults to {DEFAULT_CONFIG_PATH}."
    ),
    bbox: str = typer.Option(None, "--bbox", help="Cover this box instead of the AOI config."),
    dest: Path = typer.Option(None, "--dest", help="Where to write. Defaults to MTB_DATA_DIR/dem."),
    force: bool = typer.Option(False, "--force", help="Re-download tiles already present."),
) -> None:
    """Download the Copernicus GLO-30 tiles covering the area of interest.

    Public Cloud-Optimized GeoTIFFs on AWS Open Data: no account, no API key.
    Roughly 42 MB per one-degree tile.
    """
    settings = get_settings()
    target_dir = dest or settings.dem_dir

    area = parse_bbox(bbox) if bbox else load_aoi_config(config or DEFAULT_CONFIG_PATH).as_bbox()
    if area is None:
        msg = "No area to cover: give --bbox, or set one in the AOI config"
        raise typer.BadParameter(msg)

    tiles = tiles_for_bbox(area)
    console.print(f"Fetching {len(tiles)} DEM tile(s) into [bold]{target_dir}[/bold]")
    target_dir.mkdir(parents=True, exist_ok=True)

    absent = 0
    for tile in tiles:
        destination = target_dir / tile.filename
        if destination.exists() and not force:
            console.print(f"  [dim]have[/dim] {tile.name}")
            continue
        try:
            _download_tile(tile.url, destination)
        except MtbPlannerError as exc:
            # GLO-30 publishes land only, and a bounding box over the Channel or
            # the North Sea asks for tiles that were never made. That is the
            # normal shape of a rectangular area of interest, not a failure, so
            # it is counted and reported rather than aborting the fetch and
            # leaving the remaining tiles undownloaded.
            absent += 1
            console.print(f"  [dim]none[/dim] {tile.name} (no tile published)")
            logger.debug("No DEM tile for %s: %s", tile.name, exc)
            continue
        console.print(
            f"  [green]OK[/green] {tile.name}  ({destination.stat().st_size / 1e6:.1f} MB)"
        )

    if absent:
        console.print(
            f"{absent} of {len(tiles)} tiles are not published, which is expected "
            f"where the box covers sea."
        )
