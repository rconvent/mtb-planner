"""Health endpoint.

Deliberately never fails: a 503 from the planning endpoints is only
interpretable if this one still answers and explains why.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from mtb_planner import __version__
from mtb_planner.api.schemas.common import ApiResponse
from mtb_planner.api.schemas.health import HealthData
from mtb_planner.services.graph_registry import get_registry

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
