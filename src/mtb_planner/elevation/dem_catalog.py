"""Locating Copernicus GLO-30 tiles.

GLO-30 is published as public Cloud-Optimized GeoTIFFs on the AWS Open Data
registry: anonymous HTTPS, no account, no API key. It is TanDEM-X derived and
2021-vintage, which matters under Ardennes canopy where SRTM is noisiest.

Tiles are one degree square, named after their south-west corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mtb_planner.core.geo import BBox

#: Public bucket endpoint. No credentials, no signing.
GLO30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

#: Roughly one arcsecond at the equator; the raster's nominal resolution.
GLO30_RESOLUTION_M = 30.0


@dataclass(frozen=True, slots=True)
class DemTile:
    """One degree square of elevation, named after its south-west corner."""

    lat: int
    lon: int

    @property
    def name(self) -> str:
        """The tile's identifier, which is also its directory and file stem."""
        ns = "N" if self.lat >= 0 else "S"
        ew = "E" if self.lon >= 0 else "W"
        return f"Copernicus_DSM_COG_10_{ns}{abs(self.lat):02d}_00_{ew}{abs(self.lon):03d}_00_DEM"

    @property
    def filename(self) -> str:
        """Local file name for the tile."""
        return f"{self.name}.tif"

    @property
    def url(self) -> str:
        """Download URL. The tile is nested in a directory of its own name."""
        return f"{GLO30_BASE}/{self.name}/{self.name}.tif"

    def contains(self, lat: float, lon: float) -> bool:
        """Whether a coordinate falls in this tile."""
        return self.lat <= lat < self.lat + 1 and self.lon <= lon < self.lon + 1


def tile_for(lat: float, lon: float) -> DemTile:
    """The tile covering a coordinate."""
    return DemTile(lat=math.floor(lat), lon=math.floor(lon))


def tiles_for_bbox(bbox: BBox) -> list[DemTile]:
    """Every tile needed to cover a bounding box.

    The upper bounds are inclusive: a box ending exactly on a degree line
    still touches the tile beyond it, and a missing edge tile shows up as a
    hole in the elevation rather than an error.
    """
    return [
        DemTile(lat=lat, lon=lon)
        for lat in range(math.floor(bbox.min_lat), math.floor(bbox.max_lat) + 1)
        for lon in range(math.floor(bbox.min_lon), math.floor(bbox.max_lon) + 1)
    ]
