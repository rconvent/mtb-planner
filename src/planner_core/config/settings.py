"""Application settings.

Every tunable lives here, sourced from the environment or a `.env` file. Each
sport subclasses `Settings` with its own prefix — `MTB_` for bikes, `TRAIL_` for
running — so the two can be configured side by side in one `.env`, and only
the defaults that must differ (artifact name, port) are overridden. No module
elsewhere should hardcode a path, threshold, or timeout; import `get_settings()`
instead.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from planner_core.sport import active_sport


def _default_concurrency() -> int:
    """Leave one core free so a saturated planner cannot freeze the machine."""
    return max(1, (os.cpu_count() or 2) - 1)


class Settings(BaseSettings):
    """Immutable application configuration, shared by every sport."""

    model_config = SettingsConfigDict(
        env_prefix="PLANNER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- paths -----------------------------------------------------------
    data_dir: Path = Field(
        default=Path("data"),
        description="Root for downloaded extracts, DEM tiles, and built artifacts.",
    )
    artifact: str = Field(
        default="aoi-v1",
        description="Name of the graph artifact the server loads at startup.",
    )
    graph_subdir: str = Field(
        default="graph",
        description=(
            "Directory under the data root holding this sport's built graphs. "
            "Separate per sport, because the area-of-interest config names one "
            "artifact and a runner's build must not overwrite a rider's."
        ),
    )

    # --- server ----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    log_file: Path | None = None

    # --- routing budgets -------------------------------------------------
    route_timeout_s: float = Field(
        default=20.0,
        gt=0,
        description=(
            "Wall-clock ceiling for one plan request. The generator is anytime: on "
            "expiry it returns its best candidate so far rather than failing."
        ),
    )
    max_concurrent_plans: int = Field(default_factory=_default_concurrency, ge=1)
    route_cache_ttl_s: int = Field(default=3600, gt=0)
    route_cache_size: int = Field(default=256, gt=0)

    # --- routing tolerances ----------------------------------------------
    budget_tolerance: float = Field(
        default=0.10,
        gt=0,
        lt=1,
        description="Accepted deviation from the requested time budget.",
    )
    max_overlap_ratio: float = Field(
        default=0.25,
        gt=0,
        le=1,
        description=(
            "Maximum fraction of a loop's length that may be ridden twice. Some "
            "overlap is legitimate — trailhead access is often unavoidable."
        ),
    )
    max_snap_distance_m: float = Field(
        default=250.0,
        gt=0,
        description="Beyond this, a requested point is reported as off-network.",
    )

    # --- basemap ---------------------------------------------------------
    basemap_url: str = Field(
        default="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        description=(
            "Raster tile template drawn under the trail network, for orientation. "
            "Set to an empty string to run fully offline: the map then falls back "
            "to plain paper with the route and trails on it, which is what every "
            "other part of this project assumes."
        ),
    )
    basemap_attribution: str = Field(
        default=(
            "Basemap and trail data (c) OpenStreetMap contributors, ODbL. "
            "Tiles courtesy of the OpenStreetMap Foundation."
        ),
        description="Credit shown on the map. Required by most tile providers.",
    )
    basemap_max_zoom: int = Field(default=19, ge=1, le=24)

    @property
    def basemap_enabled(self) -> bool:
        """Whether a basemap is configured at all."""
        return bool(self.basemap_url.strip())

    @property
    def basemap_host(self) -> str:
        """Origin the tiles come from, for the content security policy.

        Derived rather than written down twice: a policy naming a different host
        from the one actually configured blocks every tile, and the map goes
        blank with only a console error to explain it.
        """
        if not self.basemap_enabled:
            return ""
        parsed = urlsplit(self.basemap_url)
        return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme else ""

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    # --- derived paths ---------------------------------------------------

    @property
    def osm_dir(self) -> Path:
        """Where downloaded `.osm.pbf` extracts live."""
        return self.data_dir / "osm"

    @property
    def dem_dir(self) -> Path:
        """Where downloaded DEM tiles live."""
        return self.data_dir / "dem"

    @property
    def graph_dir(self) -> Path:
        """Parent directory holding this sport's built graph artifacts."""
        return self.data_dir / self.graph_subdir

    @property
    def artifact_dir(self) -> Path:
        """The specific artifact this server instance loads."""
        return self.graph_dir / self.artifact


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings for the active sport.

    Cached so the `.env` file is read once. Activating a sport clears the
    cache. Tests that need different values should call
    `get_settings.cache_clear()` or override the FastAPI dependency rather than
    mutating the instance, which is frozen.
    """
    return active_sport().settings_cls()
