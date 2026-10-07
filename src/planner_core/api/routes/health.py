"""Health endpoint.

Deliberately never fails: a 503 from the planning endpoints is only
interpretable if this one still answers and explains why.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from planner_core import __version__
from planner_core.api.schemas.common import ApiResponse
from planner_core.api.schemas.health import HealthData
from planner_core.services.graph_registry import get_registry

router = APIRouter(tags=["health"])

_STARTED_AT = time.monotonic()


@router.get("/api/health", response_model=ApiResponse[HealthData])
def health() -> ApiResponse[HealthData]:
    """Report liveness and routing-graph state."""
    state = get_registry().state()
    data = HealthData(
        status="ok" if state.loaded else "degraded",
        version=__version__,
        uptime_s=round(time.monotonic() - _STARTED_AT, 3),
        graph=state,
    )
    return ApiResponse.ok(data)
