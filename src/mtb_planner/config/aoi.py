"""The area of interest a build covers.

Kept as a file rather than command line flags because it is the one piece of
configuration that must stay identical between fetching extracts and building
the graph. Passing the region list twice is how the two drift apart.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mtb_planner.core.errors import BuildError
from mtb_planner.core.geo import BBox
from mtb_planner.ingest.catalog import Region

DEFAULT_CONFIG_PATH = Path("config/aoi.toml")


class AoiConfig(BaseModel):
    """Which extracts to use, and how far to clip them."""

    model_config = ConfigDict(frozen=True)

    artifact: str = Field(min_length=1, description="Name of the artifact directory to build.")
    regions: tuple[str, ...] = Field(min_length=1, description="Geofabrik region slugs.")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description=(
            "min_lon, min_lat, max_lon, max_lat. Whole-country extracts are clipped "
            "to this; a way survives if any of its nodes falls inside."
        ),
    )
    contract: bool = Field(
        default=True, description="Merge chains that pass straight through a node."
    )

    @model_validator(mode="after")
    def _check_bbox(self) -> AoiConfig:
        if self.bbox is not None:
            min_lon, min_lat, max_lon, max_lat = self.bbox
            if min_lon >= max_lon or min_lat >= max_lat:
                msg = "bbox minimums must be smaller than its maximums"
                raise ValueError(msg)
        return self

    def as_bbox(self) -> BBox | None:
        """The clipping box, in the internal geometry type."""
        if self.bbox is None:
            return None
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return BBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)

    def extract_paths(self, osm_dir: Path) -> list[Path]:
        """Where each region's downloaded extract is expected to live."""
        return [osm_dir / Region(slug).filename for slug in self.regions]


def load_aoi_config(path: Path = DEFAULT_CONFIG_PATH) -> AoiConfig:
    """Read and validate an area-of-interest definition."""
    if not path.exists():
        msg = f"No area-of-interest config at {path}"
        raise BuildError(msg)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        msg = f"{path} is not valid TOML: {exc}"
        raise BuildError(msg) from exc

    try:
        return AoiConfig.model_validate(raw)
    except ValidationError as exc:
        msg = f"{path} is not a usable area-of-interest config: {exc}"
        raise BuildError(msg) from exc
