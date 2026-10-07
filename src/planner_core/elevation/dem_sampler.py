"""Reading elevations out of the DEM.

Bilinear, never nearest-neighbour. Nearest sampling snaps every point to a
pixel centre, so a trail running diagonally across the raster picks up a
staircase — and differencing a staircase invents grades no rider ever climbed.

Points are grouped by tile and each tile is read once. Opening a 42 MB
GeoTIFF per point would dominate a build; opening it per tile costs a second.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio

from planner_core.config.logging import get_logger
from planner_core.core.errors import DemTileMissingError
from planner_core.elevation.dem_catalog import DemTile, tile_for
from planner_core.elevation.dem_unpublished import load_unpublished

logger = get_logger(__name__)

#: Pixel coordinates from rasterio's inverse transform are measured from the
#: upper-left corner of the raster, so a pixel's centre sits half a pixel in.
_PIXEL_CENTRE_OFFSET = 0.5

#: How many missing tile names to list before summarising the rest.
_MAX_LISTED_MISSING = 5


def _bilinear(
    raster: npt.NDArray[np.float64],
    rows: npt.NDArray[np.float64],
    columns: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Interpolate between the four pixel centres around each point."""
    height, width = raster.shape

    row0 = np.clip(np.floor(rows).astype(np.int64), 0, height - 1)
    col0 = np.clip(np.floor(columns).astype(np.int64), 0, width - 1)
    row1 = np.clip(row0 + 1, 0, height - 1)
    col1 = np.clip(col0 + 1, 0, width - 1)

    row_fraction = np.clip(rows - row0, 0.0, 1.0)
    col_fraction = np.clip(columns - col0, 0.0, 1.0)

    top = raster[row0, col0] * (1 - col_fraction) + raster[row0, col1] * col_fraction
    bottom = raster[row1, col0] * (1 - col_fraction) + raster[row1, col1] * col_fraction
    return top * (1 - row_fraction) + bottom * row_fraction


class DemSampler:
    """Samples elevations from local GLO-30 tiles."""

    def __init__(self, tile_dir: Path) -> None:
        self._tile_dir = tile_dir
        self._missing: set[str] = set()

    @property
    def missing_tiles(self) -> frozenset[str]:
        """Tiles that were needed but absent."""
        return frozenset(self._missing)

    def sample(
        self, lat: npt.NDArray[np.float64], lon: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return elevations in metres, `NaN` where no tile covers the point."""
        elevations = np.full(lat.shape[0], np.nan, dtype=np.float64)
        if lat.shape[0] == 0:
            return elevations

        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (point_lat, point_lon) in enumerate(zip(lat, lon, strict=True)):
            tile = tile_for(point_lat, point_lon)
            buckets[(tile.lat, tile.lon)].append(index)

        for (tile_lat, tile_lon), indices in buckets.items():
            tile = DemTile(lat=tile_lat, lon=tile_lon)
            selection = np.asarray(indices, dtype=np.int64)
            elevations[selection] = self._sample_tile(tile, lat[selection], lon[selection])

        return elevations

    def _sample_tile(
        self,
        tile: DemTile,
        lat: npt.NDArray[np.float64],
        lon: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Bilinearly sample every point falling inside one tile."""
        path = self._tile_dir / tile.filename
        if not path.exists():
            if tile.name not in self._missing:
                self._missing.add(tile.name)
                logger.warning("Missing DEM tile %s; elevations there will be absent", tile.name)
            return np.full(lat.shape[0], np.nan, dtype=np.float64)

        with rasterio.open(path) as source:
            raster = source.read(1).astype(np.float64)
            nodata = source.nodata
            inverse = ~source.transform
            columns, rows = inverse * (lon, lat)

        if nodata is not None:
            raster[raster == nodata] = np.nan

        return _bilinear(raster, rows - _PIXEL_CENTRE_OFFSET, columns - _PIXEL_CENTRE_OFFSET)


def require_tiles(tile_dir: Path, tiles: list[DemTile]) -> None:
    """Fail before a build starts if any tile is absent.

    Silently accepting a gap yields a graph that is flat in one region, and
    flat terrain is indistinguishable from correct output until a rider
    notices the ascent figures are nonsense.

    Squares the archive never published are exempt. GLO-30 is land only, so
    an area of interest that reaches open water always asks for a few tiles
    that cannot be downloaded; demanding those makes the build impossible to
    run rather than safe. `mtb fetch dem` records which ones the archive
    refused, and only that recorded set is excused here.
    """
    unpublished = load_unpublished(tile_dir)
    missing = [
        tile.name
        for tile in tiles
        if tile.name not in unpublished and not (tile_dir / tile.filename).exists()
    ]
    if not missing:
        if unpublished:
            wanted = sum(1 for tile in tiles if tile.name in unpublished)
            if wanted:
                logger.info(
                    "%d of %d tile(s) covering this area are sea and were never published",
                    wanted,
                    len(tiles),
                )
        return

    listed = ", ".join(sorted(missing)[:_MAX_LISTED_MISSING])
    extra = len(missing) - _MAX_LISTED_MISSING
    more = f" (and {extra} more)" if extra > 0 else ""
    msg = f"Missing {len(missing)} DEM tile(s): {listed}{more}. Run `fetch dem` first."
    raise DemTileMissingError(msg)
