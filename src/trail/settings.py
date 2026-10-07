"""Settings for the trail planner, read from `TRAIL_` environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from planner_core.config.settings import Settings


class TrailSettings(Settings):
    """The shared settings, with the defaults that must differ from the bike's.

    The data directory is the same by default, so extracts and elevation tiles
    downloaded for one sport serve the other. Built graphs live apart, and the
    server takes the next port, so both planners can run side by side.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRAIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    data_dir: Path = Field(default=Path("data"))
    graph_subdir: str = Field(default="graph-trail")
    port: int = Field(default=8001, ge=1, le=65535)
