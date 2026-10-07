"""The shared response envelope and coordinate model.

Every endpoint returns `ApiResponse`, on success and on failure alike, so
clients parse one shape and never have to branch on status code first.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from planner_core.core.geo import LatLon


class ErrorDetail(BaseModel):
    """A machine-readable failure description."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Stable identifier, e.g. 'point_off_network'.")
    message: str = Field(description="Human-readable explanation.")
    detail: dict[str, Any] | None = Field(
        default=None,
        description="Structured context, e.g. the nearest routable point.",
    )


class ApiResponse[T](BaseModel):
    """Uniform envelope wrapping every response body."""

    success: bool
    data: T | None = None
    error: ErrorDetail | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, data: T, **meta: Any) -> ApiResponse[T]:
        """Build a success envelope."""
        return cls(success=True, data=data, error=None, meta=meta)

    @classmethod
    def fail(
        cls,
        code: str,
        message: str,
        detail: dict[str, Any] | None = None,
        **meta: Any,
    ) -> ApiResponse[T]:
        """Build a failure envelope."""
        return cls(
            success=False,
            data=None,
            error=ErrorDetail(code=code, message=message, detail=detail),
            meta=meta,
        )


class LatLonModel(BaseModel):
    """A WGS84 coordinate crossing the API boundary.

    Range-checked here so nothing downstream has to. Coverage-area checks are
    separate and happen against the loaded artifact's bounding box.
    """

    model_config = ConfigDict(frozen=True)

    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)

    def to_domain(self) -> LatLon:
        """Convert to the internal geometry type."""
        return LatLon(lat=self.lat, lon=self.lon)

    @classmethod
    def from_domain(cls, point: LatLon) -> LatLonModel:
        """Convert from the internal geometry type."""
        return cls(lat=point.lat, lon=point.lon)
