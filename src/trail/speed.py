"""How fast a runner covers ground.

Climbing and flat running come from the metabolic cost of running on a slope,
measured by Minetti et al. (2002) on a treadmill between −45% and +45%:

    C(i) = 155.4·i⁵ − 30.4·i⁴ − 43.3·i³ + 46.3·i² + 19.5·i + 3.6   [J/kg/m]

A runner sustains roughly constant metabolic power, so speed on any grade is
that power over the cost of a metre there. The power is calibrated from one
number a runner actually knows — their pace on flat tarmac — rather than from
a VO₂ figure they do not.

`C(i)` bottoms out near −20%, which on its own would send a runner down a
forest descent at 20 km/h. Descending is limited by footing, not energy, so
speed is capped by technicality and surface exactly as it is for a bike —
the cap applies on the flat too, because nobody runs a T3 at road pace.

Surface costs energy as well as footing. Sand is the extreme case, around 1.2
to 1.6 times the cost of firm ground (Lejeune et al., 1998); the rest sit
between that and tarmac. Technical ground costs a further share per grade:
shorter strides, line choice, and stopping to look.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from planner_core.core.units import SECONDS_PER_MINUTE
from planner_core.cost.speed import GRADE_BAND_CENTRES, TECH_BUCKETS
from planner_core.features.enums import SurfaceClass

#: Minetti's polynomial, highest power first, in J/kg/m.
MINETTI_COEFFICIENTS: tuple[float, ...] = (155.4, -30.4, -43.3, 46.3, 19.5, 3.6)

#: The polynomial was fitted between these grades. Beyond them it is
#: extrapolation, so grades are clamped first.
MINETTI_GRADE_LIMIT = 0.45

#: Energy cost relative to tarmac, by surface.
SURFACE_COST_FACTOR: dict[SurfaceClass, float] = {
    SurfaceClass.PAVED: 1.00,
    SurfaceClass.COMPACTED: 1.03,
    SurfaceClass.GRAVEL: 1.06,
    SurfaceClass.GROUND: 1.08,
    SurfaceClass.GRASS: 1.12,
    SurfaceClass.ROCK: 1.15,
    SurfaceClass.SAND_MUD: 1.35,
    SurfaceClass.UNKNOWN: 1.06,
}

#: Extra energy cost per grade of technicality.
TECH_COST_FACTOR = 0.12

#: Fastest a runner will descend at each whole grade of technicality, in m/s.
#: 5 m/s is 18 km/h, a quick but sustainable descent on a smooth track.
DESCENT_CAP_M_S: npt.NDArray[np.float64] = np.array(
    [5.0, 4.2, 3.3, 2.4, 1.6, 1.1, 0.8], dtype=np.float64
)

#: Surface also limits how hard a descent can be pushed.
DESCENT_SURFACE_FACTOR: dict[SurfaceClass, float] = {
    SurfaceClass.PAVED: 1.00,
    SurfaceClass.COMPACTED: 1.00,
    SurfaceClass.GRAVEL: 0.95,
    SurfaceClass.GROUND: 0.95,
    SurfaceClass.GRASS: 0.90,
    SurfaceClass.ROCK: 0.80,
    SurfaceClass.SAND_MUD: 0.80,
    SurfaceClass.UNKNOWN: 0.95,
}

#: Bounds on a runner's speed, in m/s. The floor is a slow power-hike up the
#: steepest band; the ceiling is a sprint nobody holds on a trail.
MIN_RUN_SPEED_M_S = 0.4
MAX_RUN_SPEED_M_S = 6.0

#: Pace on ways that need hands: scrambling, not running.
SCRAMBLE_SPEED_M_S = 0.8


def cost_of_running(grade: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Minetti's metabolic cost of running at each grade, in J/kg/m."""
    clamped = np.clip(grade, -MINETTI_GRADE_LIMIT, MINETTI_GRADE_LIMIT)
    return np.asarray(np.polyval(MINETTI_COEFFICIENTS, clamped), dtype=np.float64)


@dataclass(frozen=True)
class Runner:
    """The engine, set by the pace it holds on the flat."""

    #: Minutes per kilometre on flat tarmac. 6:00 is a comfortable long run.
    flat_pace_min_km: float = 6.0
    #: Scales the descent caps: a cautious runner brakes earlier.
    descent_skill: float = 1.0

    @property
    def flat_speed_m_s(self) -> float:
        """Speed on flat tarmac."""
        return 1000.0 / (self.flat_pace_min_km * SECONDS_PER_MINUTE)

    @property
    def metabolic_power_w_kg(self) -> float:
        """Sustained power, calibrated so flat tarmac comes out at the given pace."""
        flat_cost = float(cost_of_running(np.zeros(1))[0])
        return self.flat_speed_m_s * flat_cost

    @property
    def walk_speed_m_s(self) -> float:
        """Pace where a way can only be scrambled."""
        return SCRAMBLE_SPEED_M_S

    def speed_table(self) -> npt.NDArray[np.float64]:
        """Speed for every grade band, surface and technicality."""
        return speed_table(self)


def speed_table(runner: Runner) -> npt.NDArray[np.float64]:
    """Speed in m/s for every grade band, surface and technicality.

    Evaluated once per runner — a few hundred entries — so per-request timing
    is a lookup and a division rather than a polynomial per metre.
    """
    surfaces = range(len(SurfaceClass))
    tech = np.arange(TECH_BUCKETS, dtype=np.float64)[None, None, :]

    cost = cost_of_running(GRADE_BAND_CENTRES)[:, None, None]
    surface_cost = np.array([SURFACE_COST_FACTOR[SurfaceClass(s)] for s in surfaces])
    cost = cost * surface_cost[None, :, None] * (1.0 + TECH_COST_FACTOR * tech)

    speed = runner.metabolic_power_w_kg / cost

    surface_cap = np.array([DESCENT_SURFACE_FACTOR[SurfaceClass(s)] for s in surfaces])
    cap = (
        DESCENT_CAP_M_S[np.arange(TECH_BUCKETS)][None, None, :]
        * surface_cap[None, :, None]
        * runner.descent_skill
    )

    limited: npt.NDArray[np.float64] = np.clip(
        np.minimum(speed, cap), MIN_RUN_SPEED_M_S, MAX_RUN_SPEED_M_S
    )
    return limited
