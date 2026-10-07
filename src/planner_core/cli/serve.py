"""`serve` — run the API server.

Single worker by design: the graph is held in memory, so each additional worker
process would hold its own full copy.
"""

from __future__ import annotations

import os

import typer
import uvicorn
from rich.console import Console

from planner_core.config.settings import Settings, get_settings
from planner_core.graph.manifest import MANIFEST_FILENAME, resolve_artifact_dir
from planner_core.sport import active_sport

console = Console()


def _artifact_env() -> str:
    """The environment variable the active sport's settings read the artifact from.

    Set rather than passed as an argument, because `--reload` runs the app in a
    child process that is given the app by import string and inherits the
    environment.
    """
    prefix = active_sport().settings_cls.model_config.get("env_prefix", "")
    return f"{prefix}ARTIFACT"


def serve(
    host: str | None = typer.Option(None, help="Bind address. Defaults to the HOST setting."),
    port: int | None = typer.Option(None, help="Port. Defaults to the PORT setting."),
    artifact: str | None = typer.Option(
        None, "--artifact", help="Artifact name to serve. Defaults to the ARTIFACT setting."
    ),
    reload: bool = typer.Option(False, "--reload", help="Reload on source changes."),
) -> None:
    """Start the FastAPI server."""
    if artifact is not None:
        os.environ[_artifact_env()] = artifact
        get_settings.cache_clear()

    settings = get_settings()
    _warn_if_artifact_missing(settings)

    uvicorn.run(
        active_sport().asgi_app,
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        workers=1,
        log_level=settings.log_level.lower(),
    )


def _warn_if_artifact_missing(settings: Settings) -> None:
    """Say plainly, before uvicorn starts, that nothing can be planned.

    The server deliberately stays up without an artifact so `/api/health` can
    explain why, but that explanation is one line in a log that immediately
    scrolls under a page of request traces. Anybody who has just run `serve`
    should be told here, where they are looking.
    """
    configured = settings.artifact_dir
    if (configured / MANIFEST_FILENAME).is_file():
        return

    resolved = resolve_artifact_dir(configured)
    if resolved != configured:
        console.print(
            f"[yellow]{settings.artifact} is not built; serving "
            f"[bold]{resolved.name}[/bold], the only artifact present.[/yellow]"
        )
        return

    parent = configured.parent
    available = (
        sorted(child.name for child in parent.iterdir() if (child / MANIFEST_FILENAME).is_file())
        if parent.is_dir()
        else []
    )

    console.print(f"[yellow]No routing artifact at {configured}.[/yellow]")
    if available:
        console.print(
            f"  Several are built and none is named {settings.artifact}, so there is "
            f"nothing to choose: [bold]{', '.join(available)}[/bold]."
        )
        cli = active_sport().cli
        console.print(f"  Restart with: [bold]{cli} serve --artifact {available[0]}[/bold]")
    else:
        console.print(f"  Nothing is built yet. Run [bold]{active_sport().cli} build[/bold] first.")
