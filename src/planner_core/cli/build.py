"""`build` — turn extracts into the routing artifact the server loads."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from planner_core.cli.parsing import parse_bbox
from planner_core.config.aoi import DEFAULT_CONFIG_PATH, load_aoi_config
from planner_core.config.settings import get_settings
from planner_core.core.errors import BuildError, PlannerError
from planner_core.core.geo import BBox
from planner_core.elevation.dem_catalog import tiles_for_bbox
from planner_core.elevation.dem_sampler import DemSampler, require_tiles
from planner_core.elevation.footprint import elevation_footprint
from planner_core.graph.builder import build_graph
from planner_core.graph.manifest import Manifest, SourceExtract
from planner_core.graph.serialize import artifact_size_bytes, save_artifact
from planner_core.ingest.pbf_reader import IngestResult, read_extracts
from planner_core.sport import active_sport

console = Console()

#: Below this share of edges, most of the network is unreachable and the
#: artifact is probably not worth serving.
_HEALTHY_SCC_FRACTION = 0.97

#: Above this many seconds, report the duration in minutes instead.
_MINUTES_THRESHOLD_S = 90.0


@dataclass(frozen=True)
class _BuildInputs:
    """Everything a build needs, once flags and config are reconciled."""

    paths: list[Path]
    artifact: str
    bbox: BBox | None
    contract: bool


def _elevation_sampler(ingest: IngestResult, dem_dir: Path, enabled: bool) -> DemSampler | None:
    """Prepare DEM sampling, failing before the build if tiles are absent.

    Checked up front rather than mid-build: a gap discovered halfway through
    would already have cost the ingest, and a graph that is flat in one region
    looks perfectly valid from the outside.
    """
    if not enabled:
        console.print("[yellow]Building without elevation.[/yellow] Climb will read as zero.")
        return None

    require_tiles(dem_dir, tiles_for_bbox(elevation_footprint(ingest.ways, ingest.nodes)))
    return DemSampler(dem_dir)


def _sources(paths: list[Path]) -> tuple[SourceExtract, ...]:
    """Record what went into a build, so an artifact can be traced back."""
    return tuple(SourceExtract(name=path.name, bytes=path.stat().st_size) for path in paths)


def _resolve_inputs(
    pbf: list[Path] | None,
    config: Path | None,
    artifact: str | None,
    bbox: str | None,
    contract: bool,
) -> _BuildInputs:
    """Work out which extracts to read, and how to clip them."""
    settings = get_settings()

    if pbf:
        return _BuildInputs(
            paths=list(pbf),
            artifact=artifact or settings.artifact,
            bbox=parse_bbox(bbox),
            contract=contract,
        )

    aoi = load_aoi_config(config or DEFAULT_CONFIG_PATH)
    paths = aoi.extract_paths(settings.osm_dir)
    missing = [path for path in paths if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        msg = f"Missing extract(s): {names}. Run `{active_sport().cli} fetch osm` first."
        raise BuildError(msg)

    return _BuildInputs(
        paths=paths,
        artifact=artifact or aoi.artifact,
        bbox=parse_bbox(bbox) if bbox else aoi.as_bbox(),
        # A config that disables contraction is not overridden by the flag's
        # default; only an explicit --no-contract can turn it off further.
        contract=contract and aoi.contract,
    )


def _report(manifest: Manifest, directory: Path, elapsed_s: float) -> None:
    """Print what was built, and flag a graph that is mostly unreachable."""
    counts = manifest.counts
    reachable = counts.giant_scc_edge_fraction
    duration = (
        f"{elapsed_s / 60:.1f} min" if elapsed_s > _MINUTES_THRESHOLD_S else f"{elapsed_s:.1f}s"
    )

    table = Table(show_header=False, box=None)
    table.add_column("", style="dim")
    table.add_column("", justify="right")
    table.add_row("artifact", str(directory))
    table.add_row("built in", duration)
    table.add_row("nodes", f"{counts.nodes:,}")
    table.add_row("edges", f"{counts.edges:,}")
    table.add_row("arcs", f"{counts.allowed_arcs:,} of {counts.arcs:,} traversable")
    table.add_row("geometry", f"{counts.geom_points:,} points")
    table.add_row("reachable", f"{counts.giant_scc_edges:,} edges  ({reachable * 100:.1f}%)")
    table.add_row("contracted", "yes" if manifest.contracted else "no")
    table.add_row("total ascent", f"{manifest.total_ascent_m / 1000:,.1f} km of climb")
    for level in ("surveyed", "proxy", "estimated", "default"):
        share = manifest.tech_coverage.get(level, 0.0)
        table.add_row(f"difficulty {level}", f"{share * 100:.1f}% of length")
    table.add_row("on disk", f"{artifact_size_bytes(directory) / 1e6:.1f} MB")
    console.print(table)

    surveyed = manifest.tech_coverage.get("surveyed", 0.0)
    console.print(
        f"\n[dim]{surveyed * 100:.1f}% of network length has a surveyed "
        f"{active_sport().surveyed_tag}; the rest "
        "is inferred from surface, tracktype, smoothness and width.[/dim]"
    )

    if reachable < _HEALTHY_SCC_FRACTION:
        console.print(
            f"\n[yellow]Only {reachable * 100:.1f}% of edges are reachable within the main "
            "network.[/yellow] Snapping outside it produces 'no route found'."
        )


def build(
    pbf: list[Path] = typer.Option(
        None, "--pbf", help="Build from these extracts instead of the AOI config."
    ),
    config: Path = typer.Option(
        None, "--config", help=f"Area-of-interest config. Defaults to {DEFAULT_CONFIG_PATH}."
    ),
    artifact: str = typer.Option(None, "--artifact", help="Override the artifact name."),
    out: Path = typer.Option(None, "--out", help="Override the output directory."),
    bbox: str = typer.Option(None, "--bbox", help="Override the clip box."),
    contract: bool = typer.Option(
        True, "--contract/--no-contract", help="Merge pass-through chains."
    ),
    elevation: bool = typer.Option(
        True, "--elevation/--no-elevation", help="Sample climb from the DEM."
    ),
) -> None:
    """Build the routing artifact from downloaded extracts."""
    settings = get_settings()

    try:
        inputs = _resolve_inputs(pbf, config, artifact, bbox, contract)
        directory = out or (settings.graph_dir / inputs.artifact)

        console.print(
            f"Building [bold]{inputs.artifact}[/bold] from {len(inputs.paths)} extract(s)"
        )
        started = time.perf_counter()
        ingest = read_extracts(inputs.paths, bbox=inputs.bbox)
        sampler = _elevation_sampler(ingest, settings.dem_dir, enabled=elevation)
        store = build_graph(ingest, contract_chains=inputs.contract, sampler=sampler)
        elapsed_s = time.perf_counter() - started

        manifest = save_artifact(
            store,
            directory,
            artifact=inputs.artifact,
            contracted=inputs.contract,
            sources=_sources(inputs.paths),
        )
    except PlannerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    _report(manifest, directory, elapsed_s)
