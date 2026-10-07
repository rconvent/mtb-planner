"""How fast a rider covers ground.

Tobler's hiking function and Naismith's rule are the usual reach for terrain
speed, and both are wrong here by roughly a factor of five in the regime that
dominates a mountain-bike ride. Tobler peaks at −5% grade and tops out near
6 km/h; a bike descends a forest track at 25–35 km/h.

Climbing and flat riding come from the standard cycling power balance, solved
for speed:

    P·η = m·g·s·v + Crr·m·g·cos(atan(s))·v + ½·ρ·CdA·v³

Descending, that equation produces nonsense — the power term goes negative and
speed runs away — so descent is capped by what the trail's technicality
actually allows. You do not descend an S3 at 40 km/h regardless of gravity.

The table this produces is read by `planner_core.cost.speed`, which turns it
into per-edge times the same way for every sport.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from planner_core.core.units import (
    AIR_DENSITY_KG_M3,
    DISMOUNT_SPEED_M_S,
    GRAVITY_M_S2,
    MAX_SPEED_M_S,
    MIN_SPEED_M_S,
)
from planner_core.cost.speed import GRADE_BAND_CENTRES, TECH_BUCKETS
from planner_core.features.enums import SurfaceClass

#: Rolling resistance by surface. The spread between tarmac and mud is more
#: than a factor of ten, which is why surface classification earns its place.
ROLLING_RESISTANCE: dict[SurfaceClass, float] = {
    SurfaceClass.PAVED: 0.006,
    SurfaceClass.COMPACTED: 0.012,
    SurfaceClass.GRAVEL: 0.018,
    SurfaceClass.GROUND: 0.020,
    SurfaceClass.GRASS: 0.040,
    SurfaceClass.ROCK: 0.045,
    SurfaceClass.SAND_MUD: 0.070,
    SurfaceClass.UNKNOWN: 0.018,
}

#: Technical trail costs energy beyond rolling resistance: line choice,
#: braking, and getting back up to speed.
TECH_RESISTANCE_FACTOR = 0.10

#: Fastest a rider will descend at each whole grade of technicality, in m/s.
DESCENT_CAP_M_S: npt.NDArray[np.float64] = np.array(
    [9.0, 7.5, 5.5, 4.0, 2.8, 1.8, 1.2], dtype=np.float64
)

#: Surface also limits how hard a descent can be pushed.
DESCENT_SURFACE_FACTOR: dict[SurfaceClass, float] = {
    SurfaceClass.PAVED: 1.25,
    SurfaceClass.COMPACTED: 1.10,
    SurfaceClass.GRAVEL: 1.00,
    SurfaceClass.GROUND: 0.95,
    SurfaceClass.GRASS: 0.90,
    SurfaceClass.ROCK: 0.80,
    SurfaceClass.SAND_MUD: 0.70,
    SurfaceClass.UNKNOWN: 1.00,
}

#: Newton iterations to solve the power balance. It converges in three or
#: four from this starting point; five leaves margin.
_NEWTON_STEPS = 5
_NEWTON_START_M_S = 5.0


@dataclass(frozen=True)
class Rider:
    """The engine, and how hard it is pushed."""

    power_w: float = 200.0
    mass_kg: float = 90.0
    drag_area_m2: float = 0.45
    drivetrain_efficiency: float = 0.97
    #: Scales the descent caps: a cautious rider brakes earlier.
    descent_skill: float = 1.0

    @property
    def walk_speed_m_s(self) -> float:
        """Pushing pace, where a way may only be walked."""
        return DISMOUNT_SPEED_M_S

    def speed_table(self) -> npt.NDArray[np.float64]:
        """Speed for every grade band, surface and technicality."""
        return speed_table(self)


def solve_speed(
    grade: npt.NDArray[np.float64], rolling: npt.NDArray[np.float64], rider: Rider
) -> npt.NDArray[np.float64]:
    """Solve the power balance for speed, in m/s.

    Newton's method on `½ρ·CdA·v³ + (m·g·s + Crr·m·g·cos)·v − P·η = 0`. The
    cubic has one positive root wherever the linear coefficient is positive,
    which is every case this is asked about — descents are handled by the cap
    rather than by this equation.
    """
    drag = 0.5 * AIR_DENSITY_KG_M3 * rider.drag_area_m2
    weight = rider.mass_kg * GRAVITY_M_S2
    linear = weight * grade + rolling * weight * np.cos(np.arctan(grade))
    available = rider.power_w * rider.drivetrain_efficiency

    speed = np.full(np.broadcast(grade, rolling).shape, _NEWTON_START_M_S, dtype=np.float64)
    for _ in range(_NEWTON_STEPS):
        value = drag * speed**3 + linear * speed - available
        slope = 3.0 * drag * speed**2 + linear
        speed = np.where(slope > 1e-9, speed - value / slope, MAX_SPEED_M_S)
        speed = np.clip(speed, MIN_SPEED_M_S, MAX_SPEED_M_S)

    return speed


def speed_table(rider: Rider) -> npt.NDArray[np.float64]:
    """Speed in m/s for every grade band, surface and technicality.

    Evaluated once per rider — a few hundred entries — so per-request timing
    is a lookup and a division rather than a physics solve.
    """
    grades = GRADE_BAND_CENTRES[:, None, None]
    tech = np.arange(TECH_BUCKETS)[None, None, :]

    rolling = np.array([ROLLING_RESISTANCE[SurfaceClass(s)] for s in range(len(SurfaceClass))])
    rolling = rolling[None, :, None] * (1.0 + TECH_RESISTANCE_FACTOR * tech)

    speed = solve_speed(grades, rolling, rider)

    surface_factor = np.array(
        [DESCENT_SURFACE_FACTOR[SurfaceClass(s)] for s in range(len(SurfaceClass))]
    )
    cap = DESCENT_CAP_M_S[tech] * surface_factor[None, :, None] * rider.descent_skill

    # The cap applies on flat ground too: nobody rides flat S3 at 25 km/h.
    limited: npt.NDArray[np.float64] = np.clip(np.minimum(speed, cap), MIN_SPEED_M_S, MAX_SPEED_M_S)
    return limited
