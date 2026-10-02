"""`mtb route` — plan a point-to-point route and write it as GPX.

Prefers a built artifact, which loads in seconds. Building from extracts is
still available with `--pbf`, which is useful for a one-off area you have not
committed to an artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mtb_planner.cli.parsing import parse_bbox, parse_latlon
from mtb_planner.config.settings import get_settings
from mtb_planner.core.errors import BuildError, MtbPlannerError
from mtb_planner.cost.profiles import get_profile
from mtb_planner.export.gpx import write_gpx
from mtb_planner.graph.builder import build_graph
from mtb_planner.graph.manifest import resolve_artifact_dir
from mtb_planner.graph.serialize import load_artifact
from mtb_planner.graph.spatial_index import SpatialIndex
from mtb_planner.graph.store import GraphStore
from mtb_planner.ingest.pbf_reader import read_extracts
from mtb_planner.routing.point_to_point import PointToPointPlan, plan_point_to_point

console = Console()


@dataclass(frozen=True)
class _Loaded:
    """A graph ready to route on, and where it came from."""

    store: GraphStore
    index: SpatialIndex
    source: str
    seconds: float


def _load(pbf: list[Path] | None, artifact: Path | None, bbox: str | None) -> _Loaded:
    """Obtain a routable graph from an artifact or from raw extracts."""
    settings = get_settings()
    started = time.perf_counter()

    if pbf and artifact is not None:
        msg = "Pass either --pbf or --artifact, not both"
        raise BuildError(msg)

    if pbf:
        ingest = read_extracts(pbf, bbox=parse_bbox(bbox))
        store = build_graph(ingest)
        source = ", ".join(path.name for path in pbf)
    else:
        directory = resolve_artifact_dir(artifact or settings.artifact_dir)
        store, manifest = load_artifact(directory)
        source = manifest.artifact

    return _Loaded(
        store=store,
        index=SpatialIndex(store),
        source=source,
        seconds=time.perf_counter() - started,
    )


def _report(plan: PointToPointPlan, loaded: _Loaded, plan_s: float, written: Path) -> None:
    """Print a summary of what was loaded and what was planned."""
    stats = plan.route.stats
    table = Table(show_header=False, box=None)
    table.add_column("", style="dim")
    table.add_column("", justify="right")
    table.add_row("source", loaded.source)
    table.add_row(
        "graph",
        f"{loaded.store.n_nodes:,} nodes / {loaded.store.n_edges:,} edges "
        f"({loaded.seconds:.1f}s)",
    )
    table.add_row("distance", f"{stats.distance_km:.2f} km")
    table.add_row("climb", f"{stats.ascent_m:.0f} m up / {stats.descent_m:.0f} m down")
    table.add_row(
        "riding time",
        f"{stats.moving_time_min:.0f} min  ({stats.average_speed_km_h:.1f} km/h average)",
    )
    table.add_row("segments", f"{stats.n_edges:,}")
    if plan.within_budget is not None:
        table.add_row("time budget", "met" if plan.within_budget else "exceeded")
        table.add_row("detour spent", f"{plan.detour:.2f}")
    table.add_row("track points", f"{stats.n_points:,}")
    if stats.dismount_m > 0:
        table.add_row("on foot", f"{stats.dismount_m:.0f} m")
    table.add_row("snapped start", f"{plan.start_snap.distance_m:.0f} m to the network")
    table.add_row("snapped finish", f"{plan.end_snap.distance_m:.0f} m to the network")
    table.add_row("search", f"{plan.settled:,} nodes settled  ({plan_s * 1000:.0f} ms)")
    table.add_row("written", str(written))
    console.print(table)
    console.print(
        "\n[dim]Routes come from OpenStreetMap, which does not fully model access law. "
        "Check local signage before riding.[/dim]"
    )


def route(
    start: str = typer.Option(..., "--from", help="Start as 'lat,lon' in decimal degrees."),
    end: str = typer.Option(..., "--to", help="Finish as 'lat,lon' in decimal degrees."),
    out: Path = typer.Option(Path("out/ride.gpx"), "--out", help="Where to write the GPX."),
    artifact: Path = typer.Option(
        None, "--artifact", help="Artifact directory. Defaults to the configured one."
    ),
    pbf: list[Path] = typer.Option(
        None, "--pbf", help="Build from these extracts instead of loading an artifact."
    ),
    bbox: str = typer.Option(
        None, "--bbox", help="Clip to 'min_lon,min_lat,max_lon,max_lat'. Only with --pbf."
    ),
    name: str = typer.Option("MTB route", "--name", help="Track name recorded in the GPX."),
    profile: str = typer.Option("flow", "--profile", help="Ride profile: xc, flow, tech, gravity."),
    minutes: int = typer.Option(
        None,
        "--minutes",
        help=(
            "Time available, in minutes. Slack rather than a target: the route still "
            "goes where you asked, and time left over after the quickest way there is "
            "spent on better ground."
        ),
    ),
) -> None:
    """Plan a route between two points and write it as a GPX track."""
    settings = get_settings()
    start_point = parse_latlon(start, "--from")
    end_point = parse_latlon(end, "--to")

    try:
        ride_profile = get_profile(profile)
        loaded = _load(pbf, artifact, bbox)

        began = time.perf_counter()
        plan = plan_point_to_point(
            loaded.store,
            loaded.index,
            start_point,
            end_point,
            max_snap_distance_m=settings.max_snap_distance_m,
            profile=ride_profile,
            coverage=loaded.store.bbox(),
            budget_s=None if minutes is None else float(minutes) * 60.0,
        )
        plan_s = time.perf_counter() - began
    except MtbPlannerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    written = write_gpx(
        plan.route,
        out,
        name=name,
        description=f"{ride_profile.name} profile, planned from {loaded.source}",
    )
    _report(plan, loaded, plan_s, written)
