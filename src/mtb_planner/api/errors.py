"""Maps the domain error vocabulary onto HTTP responses.

Registered on the app so no route handler ever builds an error body by hand,
and so failures come back in the same envelope as successes.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mtb_planner.api.schemas.common import ApiResponse
from mtb_planner.config.logging import get_logger
from mtb_planner.core.errors import (
    ArtifactError,
    GraphNotLoadedError,
    MtbPlannerError,
    NoRouteFoundError,
    PointOffNetworkError,
    PointOutsideCoverageError,
    RouteExpiredError,
    ServerBusyError,
    UnknownProfileError,
)

logger = get_logger(__name__)

#: Domain exception -> (HTTP status, stable error code).
_STATUS_MAP: dict[type[MtbPlannerError], tuple[int, str]] = {
    GraphNotLoadedError: (status.HTTP_503_SERVICE_UNAVAILABLE, "graph_not_loaded"),
    ArtifactError: (status.HTTP_503_SERVICE_UNAVAILABLE, "artifact_unavailable"),
    ServerBusyError: (status.HTTP_503_SERVICE_UNAVAILABLE, "server_busy"),
    PointOutsideCoverageError: (status.HTTP_422_UNPROCESSABLE_CONTENT, "point_outside_coverage"),
    PointOffNetworkError: (status.HTTP_422_UNPROCESSABLE_CONTENT, "point_off_network"),
    UnknownProfileError: (status.HTTP_422_UNPROCESSABLE_CONTENT, "unknown_profile"),
    NoRouteFoundError: (status.HTTP_422_UNPROCESSABLE_CONTENT, "no_route_found"),
    RouteExpiredError: (status.HTTP_410_GONE, "route_expired"),
}


def _detail_for(exc: MtbPlannerError) -> dict[str, Any] | None:
    """Extract structured context the client can act on."""
    if isinstance(exc, PointOffNetworkError):
        return {
            "requested": {"lat": exc.lat, "lon": exc.lon},
            "nearest": (
                None
                if exc.nearest_lat is None
                else {"lat": exc.nearest_lat, "lon": exc.nearest_lon}
            ),
            "distance_m": exc.distance_m,
        }
    if isinstance(exc, PointOutsideCoverageError):
        return {"requested": {"lat": exc.lat, "lon": exc.lon}}
    if isinstance(exc, NoRouteFoundError):
        return {"reason": exc.reason}
    return None


def _resolve(exc: MtbPlannerError) -> tuple[int, str]:
    """Find the most specific mapping for an exception type."""
    for exc_type, mapped in _STATUS_MAP.items():
        if isinstance(exc, exc_type):
            return mapped
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error"


def register_error_handlers(app: FastAPI) -> None:
    """Attach every exception handler to `app`."""

    @app.exception_handler(MtbPlannerError)
    async def _domain_error(_: Request, exc: MtbPlannerError) -> JSONResponse:
        http_status, code = _resolve(exc)
        # A traceback is for something we did not anticipate. Everything in the
        # status map was anticipated, including the 503s: a missing artifact is
        # a diagnosed configuration problem, and dumping forty lines of stack on
        # every request for it buries the one line that says what to do.
        if code == "internal_error":
            logger.exception("Unhandled domain error", exc_info=exc)
        elif http_status >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            logger.warning("%s: %s", code, exc)
        else:
            logger.info("%s: %s", code, exc)
        body = ApiResponse[None].fail(code=code, message=str(exc), detail=_detail_for(exc))
        return JSONResponse(status_code=http_status, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        body = ApiResponse[None].fail(
            code="validation_error",
            message="Request failed validation",
            detail={"errors": jsonable_encoder(exc.errors())},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=body.model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Log the detail, tell the client nothing that could leak internals.
        logger.exception("Unhandled exception", exc_info=exc)
        body = ApiResponse[None].fail(
            code="internal_error",
            message="An unexpected error occurred",
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body.model_dump(),
        )
