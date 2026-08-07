"""Sampling the DEM along every edge.

Edges are densified to an even spacing before sampling, because the profile
maths downstream assumes evenly spaced samples — a polyline with one point
every 300 m and another every 4 m would weight the grade histogram by how
finely somebody happened to trace the way.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mtb_planner.config.logging import get_logger
from mtb_planner.core.geo import densify
from mtb_planner.elevation.dem_sampler import DemSampler
from mtb_planner.elevation.profile import (
    SAMPLE_SPACING_M,
    EdgeElevation,
    elevate_edges,
    interpolate_structures,
)

logger = get_logger(__name__)

#: How many missing tile names to name before summarising.
_MAX_LISTED_MISSING = 3


def _densified(
    geom: npt.NDArray[np.float64],
    geom_offsets: npt.NDArray[np.int64],
    spacing_m: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Resample every edge's polyline to an even spacing."""
    chunks: list[npt.NDArray[np.float64]] = []
    counts = np.empty(geom_offsets.shape[0] - 1, dtype=np.int64)

    for edge in range(geom_offsets.shape[0] - 1):
        start, stop = int(geom_offsets[edge]), int(geom_offsets[edge + 1])
        points = densify(geom[start:stop], spacing_m)
        chunks.append(points)
        counts[edge] = points.shape[0]

    return np.vstack(chunks), np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def sample_edges(
    geom: npt.NDArray[np.float64],
    geom_offsets: npt.NDArray[np.int64],
    is_structure: npt.NDArray[np.bool_],
    sampler: DemSampler,
    spacing_m: float = SAMPLE_SPACING_M,
) -> EdgeElevation:
    """Read elevation along every edge and reduce it to climb and grades."""
    points, offsets = _densified(geom, geom_offsets, spacing_m)
    logger.info(
        "Sampling %d elevation points across %d edges", points.shape[0], is_structure.shape[0]
    )

    elevations = sampler.sample(points[:, 0], points[:, 1])
    elevations = interpolate_structures(elevations, offsets, is_structure)

    if sampler.missing_tiles:
        logger.warning(
            "Elevation incomplete: %d tile(s) absent (%s)",
            len(sampler.missing_tiles),
            ", ".join(sorted(sampler.missing_tiles)[:_MAX_LISTED_MISSING]),
        )

    return elevate_edges(elevations, offsets, spacing_m)
