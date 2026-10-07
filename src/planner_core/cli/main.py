"""CLI root, built once per sport.

`mtb` and `trail` are the same commands over a different `Sport`. Each sport
package calls `build_cli` with its own, which activates it before any
subcommand runs, so every command reads the right settings, profiles and
artifact.
"""

from __future__ import annotations

import typer

from planner_core import __version__
from planner_core.cli import build as build_cmd
from planner_core.cli import fetch as fetch_cmd
from planner_core.cli import inspect as inspect_cmd
from planner_core.cli import route as route_cmd
from planner_core.cli import serve as serve_cmd
from planner_core.config.logging import setup_logging
from planner_core.config.settings import get_settings
from planner_core.sport import Sport, activate


def build_cli(sport: Sport) -> typer.Typer:
    """The command line for one sport."""
    activate(sport)

    app = typer.Typer(
        name=sport.cli,
        help=f"{sport.copy.title}: build the routing graph and serve routes.",
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
            typer.echo(f"{sport.cli} {__version__}")
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
        activate(sport)
        settings = get_settings()
        setup_logging(level=settings.log_level, log_file=settings.log_file)

    return app
