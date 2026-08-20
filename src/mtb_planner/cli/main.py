"""CLI root.

Subcommands are added as their phases land; `build` is still to come.
"""

from __future__ import annotations

import typer

from mtb_planner import __version__
from mtb_planner.cli import build as build_cmd
from mtb_planner.cli import fetch as fetch_cmd
from mtb_planner.cli import inspect as inspect_cmd
from mtb_planner.cli import route as route_cmd
from mtb_planner.cli import serve as serve_cmd
from mtb_planner.config.logging import setup_logging
from mtb_planner.config.settings import get_settings

app = typer.Typer(
    name="mtb",
    help="Mountain bike route planner: build the routing graph and serve routes.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(fetch_cmd.app, name="fetch")
app.command(name="build")(build_cmd.build)
app.command(name="inspect")(inspect_cmd.inspect)
app.command(name="route")(route_cmd.route)
app.command(name="serve")(serve_cmd.serve)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"mtb-planner {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(  # noqa: ARG001
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Configure logging before any subcommand runs."""
    settings = get_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)


if __name__ == "__main__":
    app()
