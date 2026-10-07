"""Health endpoint models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GraphState(BaseModel):
    """What the server knows about the currently loaded routing graph."""

    model_config = ConfigDict(frozen=True)

    loaded: bool
    artifact: str | None = None
    schema_version: int | None = None
    cost_model_version: int | None = None
    nodes: int | None = None
    edges: int | None = None
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="min_lon, min_lat, max_lon, max_lat.",
    )
    loaded_at: str | None = Field(
        default=None,
        description="UTC ISO-8601 timestamp, e.g. '2026-08-10T12:03:11Z'.",
    )
    error: str | None = Field(
        default=None,
        description="Why the graph is not loaded, when it is not.",
    )


class HealthData(BaseModel):
    """Liveness plus enough state to explain a 503 without reading the logs."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(description="'ok' when routing is available, else 'degraded'.")
    version: str
    uptime_s: float
    graph: GraphState
