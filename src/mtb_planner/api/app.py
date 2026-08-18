"""FastAPI application factory.

The graph is loaded once in the lifespan handler and published to the
registry. A load failure is recorded, not raised: the process stays up so
`/api/health` can report what went wrong.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mtb_planner import __version__
from mtb_planner.api.errors import register_error_handlers
from mtb_planner.api.routes import (
    health,
    loop,
    network,
    pages,
    planned,
    point_to_point,
    profiles,
    style,
)
from mtb_planner.config.logging import get_logger, setup_logging
from mtb_planner.config.settings import Settings, get_settings
from mtb_planner.core.errors import ArtifactError
from mtb_planner.graph.manifest import resolve_artifact_dir
from mtb_planner.graph.serialize import load_artifact
from mtb_planner.graph.spatial_index import SpatialIndex
from mtb_planner.routing.kernels import warm_up
from mtb_planner.services.graph_registry import LoadedGraph, get_registry

logger = get_logger(__name__)


def _load_graph(settings: Settings) -> None:
    """Load the routing artifact into the registry.

    A missing or incompatible artifact is recorded, not raised. The process
    stays up so `/api/health` can report what went wrong; dying here would
    take down the only endpoint able to explain the failure.
    """
    registry = get_registry()
    started = time.perf_counter()

    directory = resolve_artifact_dir(settings.artifact_dir)
    if directory != settings.artifact_dir:
        logger.warning(
            "Configured artifact %s is not built; serving the only one present: %s",
            settings.artifact,
            directory.name,
        )

    try:
        store, manifest = load_artifact(directory)
        index = SpatialIndex(store)
    except ArtifactError as exc:
        # The message already names the artifacts that do exist and how to
        # point at one; see `graph.manifest.read_manifest`.
        logger.warning("Routing unavailable: %s", exc)
        registry.set_error(str(exc))
        return

    # Compile the search kernels now. Numba's first call costs seconds, and
    # with a request timeout in place the rider who paid it would just see a
    # failure.
    warm_up()

    registry.set_graph(LoadedGraph.from_artifact(store, index, manifest))
    logger.info(
        "Routing ready: %s (%d nodes, %d edges) in %.1fs",
        manifest.artifact,
        store.n_nodes,
        store.n_edges,
        time.perf_counter() - started,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Configure logging and load the graph, once per process."""
    settings = get_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)
    logger.info("Starting mtb-planner %s", __version__)
    _load_graph(settings)
    yield
    get_registry().clear()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Build the application. One factory so tests get a fresh instance."""
    app = FastAPI(
        title="MTB Route Planner",
        version=__version__,
        summary="Time-budgeted mountain bike loops and point-to-point routes.",
        lifespan=lifespan,
    )
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(profiles.router)
    app.include_router(point_to_point.router)
    app.include_router(loop.router)
    app.include_router(network.router)
    app.include_router(style.router)
    # Last: its `{route_id}` matches any single path segment, so every more
    # specific route under /api/routes must be declared ahead of it.
    app.include_router(planned.router)

    # The interface and its assets. Mounted rather than routed one file at a
    # time, and last, so nothing here can shadow an API path.
    app.mount(
        "/static",
        StaticFiles(directory=str(pages.STATIC_ROOT)),
        name="static",
    )
    app.include_router(pages.router)
    return app


app = create_app()
