"""`mtb serve` — run the API server.

Single worker by design: the graph is held in memory, so each additional worker
process would hold its own full copy.
"""

from __future__ import annotations

import os

import typer
import uvicorn
from rich.console import Console

from mtb_planner.config.settings import Settings, get_settings
from mtb_planner.graph.manifest import MANIFEST_FILENAME, resolve_artifact_dir

console = Console()

#: The environment variable the settings layer reads. Set here rather than
#: passed as an argument, because `--reload` runs the app in a child process
#: that is given the app by import string and inherits the environment.
_ARTIFACT_ENV = "MTB_ARTIFACT"


def serve(
    host: str | None = typer.Option(None, help="Bind address. Defaults to MTB_HOST."),
    port: int | None = typer.Option(None, help="Port. Defaults to MTB_PORT."),
    artifact: str | None = typer.Option(
        None, "--artifact", help="Artifact name to serve. Defaults to MTB_ARTIFACT."
    ),
    reload: bool = typer.Option(False, "--reload", help="Reload on source changes."),
) -> None:
    """Start the FastAPI server."""
    if artifact is not None:
        os.environ[_ARTIFACT_ENV] = artifact
        get_settings.cache_clear()

    settings = get_settings()
    _warn_if_artifact_missing(settings)

    uvicorn.run(
        "mtb_planner.api.app:app",
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
    scrolls under a page of request traces. Anybody who has just run `mtb serve`
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
        console.print(f"  Restart with: [bold]mtb serve --artifact {available[0]}[/bold]")
    else:
        console.print("  Nothing is built yet. Run [bold]mtb build[/bold] first.")
