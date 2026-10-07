"""Turning a speed model into traversal times.

Each sport owns its speed model — a cycling power balance, a metabolic cost of
running — and evaluates it once into a table indexed by grade band, surface and
technicality. This module only reads that table, so it is the same for every
sport.

Time is computed from each edge's grade histogram rather than its mean grade.
An edge averaging 0% may be +12% then −12%, and averaging hides exactly the
climb the model exists to charge for.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from planner_core.elevation.profile import GRADE_BANDS
from planner_core.features.enums import TECH_SCALE_MAX
from planner_core.sport import Access

#: Representative grade for each band, used to evaluate a speed model once
#: per band rather than once per metre of trail. The outer bands are open, so
#: they take a plausible interior value rather than an infinite one.
GRADE_BAND_CENTRES: npt.NDArray[np.float64] = np.array(
    [-0.30, -0.215, -0.15, -0.10, -0.06, -0.025, 0.0, 0.025, 0.06, 0.10, 0.15, 0.215, 0.30, 0.45],
    dtype=np.float64,
)

#: Whole technicality grades a speed table is built for.
TECH_BUCKETS = int(TECH_SCALE_MAX) + 1


def _time_for(
    bands: npt.NDArray[np.float64],
    surface: npt.NDArray[np.int64],
    down_bucket: npt.NDArray[np.int64],
    up_bucket: npt.NDArray[np.int64],
    table: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Sum band length over band speed, choosing technicality by direction."""
    # Bands below the midpoint are descending, so they read the downhill
    # technicality; the rest read the uphill figure.
    midpoint = GRADE_BANDS // 2
    bucket = np.where(
        np.arange(GRADE_BANDS)[None, :] < midpoint, down_bucket[:, None], up_bucket[:, None]
    )
    speeds = table[np.arange(GRADE_BANDS)[None, :], surface[:, None], bucket]
    return np.asarray((bands / speeds).sum(axis=1))


def edge_times(
    grade_hist: npt.NDArray[np.int16],
    surface_class: npt.NDArray[np.uint8],
    tech: npt.NDArray[np.float32],
    tech_up: npt.NDArray[np.float32],
    access: npt.NDArray[np.uint8],
    lengths: npt.NDArray[np.float64],
    table: npt.NDArray[np.float64],
    walk_speed_m_s: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Traversal time in seconds, in each direction along every edge.

    Travelling an edge backwards reverses its grade bands — what climbed now
    descends — and uses the uphill technicality, because a rooty descent is
    a different proposition climbed.
    """
    surface = surface_class.astype(np.int64)
    down_bucket = np.clip(np.rint(tech), 0, TECH_BUCKETS - 1).astype(np.int64)
    up_bucket = np.clip(np.rint(tech_up), 0, TECH_BUCKETS - 1).astype(np.int64)

    bands = grade_hist.astype(np.float64)
    forward = _time_for(bands, surface, down_bucket, up_bucket, table)
    backward = _time_for(bands[:, ::-1], surface, down_bucket, up_bucket, table)

    # An edge built without elevation still takes time to cover.
    flat = bands.sum(axis=1) <= 0.0
    fallback = lengths / table[GRADE_BANDS // 2, surface, down_bucket]
    forward = np.where(flat, fallback, forward)
    backward = np.where(flat, fallback, backward)

    on_foot = access == Access.WALK
    walking = lengths / walk_speed_m_s
    return np.where(on_foot, walking, forward), np.where(on_foot, walking, backward)
