"""Settings for the bike planner, read from `MTB_` environment variables."""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from planner_core.config.settings import Settings


class MtbSettings(Settings):
    """The shared settings, under the prefix the bike planner always used."""

    model_config = SettingsConfigDict(
        env_prefix="MTB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )
