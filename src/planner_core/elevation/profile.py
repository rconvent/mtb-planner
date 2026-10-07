"""Turning sampled elevations into climb.

Three corrections stand between a raw raster read and a number worth showing
a rider, and none is optional:

* **Smoothing.** GLO-30 carries real sensor noise, and differencing noise
  produces climb. A moving average over roughly 50 m removes it without
  flattening anything a bike can feel.
* **A dead-band.** Even smoothed, tiny oscillations accumulate over thousands
  of samples. Steps below 0.8 m are discarded outright.
* **Bridges and tunnels.** A raster sampled across a viaduct returns the
  valley floor; through a tunnel, the hillside above it. Those are
  interpolated between their endpoints instead of read.

Grade is stored as a histogram of length per band rather than a mean. An edge
averaging 0% may be +12% then −12%, and averaging hides exactly the climb the
time model needs to charge for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from planner_core.config.logging import get_logger

logger = get_logger(__name__)

#: Upper bounds of each grade band, as a fraction. Fourteen bands, chosen so
#: the ones a rider actually notices are narrow and the extremes are wide.
GRADE_BAND_EDGES: tuple[float, ...] = (
    -0.25,
    -0.18,
    -0.12,
    -0.08,
    -0.04,
    -0.01,
    0.01,
    0.04,
    0.08,
    0.12,
    0.18,
    0.25,
    0.35,
)

#: Number of bands the edges above define.
GRADE_BANDS = len(GRADE_BAND_EDGES) + 1

#: Sample spacing along an edge. GLO-30 is about 30 m per pixel, so this reads
#: roughly every pixel; finer spacing samples the same value twice and adds
#: only noise.
SAMPLE_SPACING_M = 25.0

#: Width of the moving average, in metres.
SMOOTHING_WINDOW_M = 50.0

#: Steps smaller than this are treated as flat. Below roughly a metre the
#: raster cannot tell a rise from its own noise.
DEAD_BAND_M = 0.8

#: Grades beyond this are raster artefacts rather than trail, and would
#: otherwise dominate the histogram's outer bands.
MAX_PLAUSIBLE_GRADE = 0.6

#: Fewer samples than this describe no slope at all.
_MIN_SAMPLES = 2


@dataclass(frozen=True)
class EdgeElevation:
    """Climb, descent and grade distribution for every edge."""

    ascent_m: npt.NDArray[np.float32]
    descent_m: npt.NDArray[np.float32]
    grade_max: npt.NDArray[np.float32]
    grade_min: npt.NDArray[np.float32]
    grade_hist: npt.NDArray[np.int16]

    def __len__(self) -> int:
        """Number of edges described."""
        return int(self.ascent_m.shape[0])

    @property
    def total_ascent_m(self) -> float:
        """Summed climb across every edge."""
        return float(self.ascent_m.sum())

    def select(self, index: npt.NDArray[np.intp]) -> EdgeElevation:
        """Return the elevation of a subset of edges, in the given order."""
        return EdgeElevation(
            ascent_m=self.ascent_m[index],
            descent_m=self.descent_m[index],
            grade_max=self.grade_max[index],
            grade_min=self.grade_min[index],
            grade_hist=self.grade_hist[index],
        )

    @classmethod
    def empty(cls, count: int) -> EdgeElevation:
        """Flat elevation for every edge, used when no DEM is available."""
        return cls(
            ascent_m=np.zeros(count, dtype=np.float32),
            descent_m=np.zeros(count, dtype=np.float32),
            grade_max=np.zeros(count, dtype=np.float32),
            grade_min=np.zeros(count, dtype=np.float32),
            grade_hist=np.zeros((count, GRADE_BANDS), dtype=np.int16),
        )


def smooth(elevations: npt.NDArray[np.float64], spacing_m: float) -> npt.NDArray[np.float64]:
    """Moving average along a profile, in metres of window rather than samples."""
    window = max(1, round(SMOOTHING_WINDOW_M / max(spacing_m, 1e-6)))
    if window <= 1 or elevations.shape[0] <= window:
        return elevations

    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(elevations, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def profile_statistics(
    elevations: npt.NDArray[np.float64], spacing_m: float
) -> tuple[float, float, float, float, npt.NDArray[np.float64]]:
    """Reduce one edge's profile to climb, descent, extremes and band lengths.

    `elevations` must already be evenly spaced at `spacing_m`.
    """
    bands = np.zeros(GRADE_BANDS, dtype=np.float64)
    if elevations.shape[0] < _MIN_SAMPLES or not np.isfinite(elevations).all():
        return 0.0, 0.0, 0.0, 0.0, bands

    smoothed = smooth(elevations, spacing_m)
    steps = np.diff(smoothed)

    # Anything below the dead-band is noise, not terrain.
    steps[np.abs(steps) < DEAD_BAND_M] = 0.0

    grades = np.clip(steps / spacing_m, -MAX_PLAUSIBLE_GRADE, MAX_PLAUSIBLE_GRADE)
    np.add.at(bands, np.digitize(grades, GRADE_BAND_EDGES), spacing_m)

    ascent = float(steps[steps > 0].sum())
    descent = float(-steps[steps < 0].sum())
    return ascent, descent, float(grades.max()), float(grades.min()), bands


def interpolate_structures(
    elevations: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.int64],
    is_structure: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Replace bridge and tunnel profiles with a straight line.

    Without this every viaduct reads as a descent into the valley it spans and
    a climb back out, and a long tunnel reads as the hill above it.
    """
    adjusted = elevations.copy()
    for edge in np.flatnonzero(is_structure):
        start, stop = int(offsets[edge]), int(offsets[edge + 1])
        if stop - start < _MIN_SAMPLES:
            continue
        first, last = adjusted[start], adjusted[stop - 1]
        if not (np.isfinite(first) and np.isfinite(last)):
            continue
        adjusted[start:stop] = np.linspace(first, last, stop - start)
    return adjusted


def elevate_edges(
    elevations: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.int64],
    spacing_m: float = SAMPLE_SPACING_M,
) -> EdgeElevation:
    """Reduce every edge's sampled profile to its elevation summary.

    `offsets` marks where each edge's samples begin, in the compressed-row
    style used throughout. Edges whose samples are incomplete — outside the
    DEM's coverage — are reported flat rather than guessed at.
    """
    count = offsets.shape[0] - 1
    result = EdgeElevation.empty(count)
    unsampled = 0

    for edge in range(count):
        start, stop = int(offsets[edge]), int(offsets[edge + 1])
        span = elevations[start:stop]
        ascent, descent, high, low, bands = profile_statistics(span, spacing_m)
        if stop - start >= _MIN_SAMPLES and not np.isfinite(span).all():
            unsampled += 1

        result.ascent_m[edge] = ascent
        result.descent_m[edge] = descent
        result.grade_max[edge] = high
        result.grade_min[edge] = low
        result.grade_hist[edge] = np.rint(bands).astype(np.int16)

    if unsampled:
        logger.warning(
            "%d edge(s) fell outside the elevation model and are recorded as flat", unsampled
        )
    logger.info("Elevation: %.0f m of ascent across %d edges", result.total_ascent_m, count)
    return result
