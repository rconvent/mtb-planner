"""`mtb inspect` — report what is in an extract or a built artifact.

For extracts the number that matters is `mtb:scale` coverage. The difficulty
model is built on the assumption that explicit technicality tagging is rare,
and that assumption should be measured against real data before the model is
tuned rather than after.

For artifacts the number that matters is how much of the network is reachable.
Snapping outside the main component produces "no route found", so an artifact
with poor connectivity fails in a way that looks like a routing bug.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mtb_planner.core.errors import MtbPlannerError
from mtb_planner.graph.manifest import Manifest, read_manifest
from mtb_planner.graph.serialize import artifact_size_bytes
from mtb_planner.ingest.stats import collect_tag_stats, format_tag_stats

console = Console()

#: Below this, the difficulty model is mostly inference and the operator
#: should be told so explicitly rather than discovering it later.
_LOW_COVERAGE = 0.10

#: Below this share of edges, most of the network cannot be routed on.
_HEALTHY_SCC_FRACTION = 0.97


def _artifact_rows(manifest: Manifest, directory: Path) -> list[tuple[str, str]]:
    """Headline facts about a built artifact."""
    counts = manifest.counts
    min_lon, min_lat, max_lon, max_lat = manifest.bbox
    return [
        ("artifact", manifest.artifact),
        ("built at", manifest.built_at),
        ("planner version", manifest.planner_version),
        ("schema / cost model", f"v{manifest.schema_version} / v{manifest.cost_model_version}"),
        ("compatible", "yes" if manifest.is_compatible() else "NO"),
        ("contracted", "yes" if manifest.contracted else "no"),
        ("sources", ", ".join(s.name for s in manifest.sources) or "-"),
        ("nodes", f"{counts.nodes:,}"),
        ("edges", f"{counts.edges:,}"),
        ("arcs", f"{counts.allowed_arcs:,} of {counts.arcs:,} traversable"),
        ("geometry", f"{counts.geom_points:,} points"),
        (
            "reachable",
            f"{counts.giant_scc_edges:,} edges  ({counts.giant_scc_edge_fraction * 100:.1f}%)",
        ),
        ("bbox", f"{min_lat:.3f}..{max_lat:.3f} N, {min_lon:.3f}..{max_lon:.3f} E"),
        *(
            (f"difficulty {level}", f"{manifest.tech_coverage.get(level, 0.0) * 100:.1f}%")
            for level in ("surveyed", "proxy", "estimated", "default")
        ),
        ("on disk", f"{artifact_size_bytes(directory) / 1e6:.1f} MB"),
    ]


def _inspect_artifact(directory: Path, as_json: bool) -> None:
    """Report on a built artifact directory."""
    manifest = read_manifest(directory)

    if as_json:
        typer.echo(manifest.model_dump_json(indent=2))
        return

    table = Table(title="Routing artifact", show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for label, value in _artifact_rows(manifest, directory):
        table.add_row(label, value)
    console.print(table)

    if not manifest.is_compatible():
        console.print(f"\n[red]{manifest.incompatibility()}[/red] Rebuild with `mtb build`.")
    elif manifest.counts.giant_scc_edge_fraction < _HEALTHY_SCC_FRACTION:
        fraction = manifest.counts.giant_scc_edge_fraction * 100
        console.print(
            f"\n[yellow]Only {fraction:.1f}% of edges are reachable within the main "
            "network.[/yellow]"
        )


def _inspect_extracts(paths: list[Path], as_json: bool, top: int) -> None:
    """Report tag coverage across one or more extracts."""
    stats = collect_tag_stats(paths)

    if as_json:
        typer.echo(json.dumps(stats.to_dict(), indent=2))
        return

    table = Table(title="Tag coverage", show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for label, value in format_tag_stats(stats, top_n=top):
        table.add_row(label, value)
    console.print(table)

    coverage = stats.mtb_scale_coverage
    if coverage < _LOW_COVERAGE:
        console.print(
            f"\n[yellow]Only {coverage * 100:.2f}% of trail ways carry mtb:scale.[/yellow] "
            "Difficulty for the rest has to be inferred from surface, tracktype, "
            "smoothness and width."
        )


def inspect(
    target: list[Path] = typer.Argument(
        ..., help="One or more .osm.pbf extracts, or an artifact directory."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the full report as JSON."),
    top: int = typer.Option(12, "--top", help="How many highway classes to list."),
) -> None:
    """Report on extracts or on a built routing artifact."""
    try:
        if len(target) == 1 and target[0].is_dir():
            _inspect_artifact(target[0], as_json)
        else:
            _inspect_extracts(target, as_json, top)
    except MtbPlannerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
