"""The running speed model."""

from __future__ import annotations

import numpy as np
import pytest

from planner_core.cost.speed import GRADE_BAND_CENTRES, TECH_BUCKETS
from planner_core.elevation.profile import GRADE_BANDS
from planner_core.features.enums import SurfaceClass
from trail.speed import (
    MAX_RUN_SPEED_M_S,
    MIN_RUN_SPEED_M_S,
    Runner,
    cost_of_running,
    speed_table,
)

pytestmark = pytest.mark.unit

FLAT = int(np.flatnonzero(GRADE_BAND_CENTRES == 0.0)[0])


def test_the_table_has_the_shape_the_timing_reads():
    assert speed_table(Runner()).shape == (GRADE_BANDS, len(SurfaceClass), TECH_BUCKETS)


def test_flat_tarmac_comes_out_at_the_given_pace():
    for pace in (4.0, 6.0, 8.0):
        speed = speed_table(Runner(flat_pace_min_km=pace))[FLAT, SurfaceClass.PAVED, 0]

        assert speed == pytest.approx(1000.0 / (pace * 60.0))


def test_minettis_curve_has_its_minimum_downhill():
    grades = np.linspace(-0.45, 0.45, 91)
    cheapest = grades[np.argmin(cost_of_running(grades))]

    assert -0.25 < cheapest < -0.15


def test_climbing_is_slower_the_steeper_it_gets():
    table = speed_table(Runner())
    uphill = table[FLAT:, SurfaceClass.GROUND, 1]

    assert np.all(np.diff(uphill) < 0)


def test_a_steep_climb_is_a_power_hike():
    steepest = speed_table(Runner())[-1, SurfaceClass.GROUND, 1] * 3.6

    assert 1.0 < steepest < 4.0


def test_descents_are_capped_by_footing_not_energy():
    table = speed_table(Runner())
    smooth = table[:FLAT, SurfaceClass.PAVED, 0].max()
    technical = table[:FLAT, SurfaceClass.ROCK, 4].max()

    # Downhill on Minetti alone would be well over 20 km/h.
    assert smooth * 3.6 <= 18.5
    assert technical < smooth / 2


def test_rough_ground_is_slower_than_tarmac():
    table = speed_table(Runner())

    assert table[FLAT, SurfaceClass.SAND_MUD, 0] < table[FLAT, SurfaceClass.PAVED, 0]
    assert table[FLAT, SurfaceClass.GROUND, 3] < table[FLAT, SurfaceClass.GROUND, 0]


def test_a_faster_runner_is_never_slower():
    quick = speed_table(Runner(flat_pace_min_km=4.5))
    steady = speed_table(Runner(flat_pace_min_km=7.0))

    assert np.all(quick >= steady)


def test_every_speed_stays_within_bounds():
    for runner in (Runner(flat_pace_min_km=3.5), Runner(flat_pace_min_km=10.0)):
        table = runner.speed_table()

        assert table.min() >= MIN_RUN_SPEED_M_S
        assert table.max() <= MAX_RUN_SPEED_M_S


def test_scrambling_is_slower_than_running():
    runner = Runner()

    assert runner.walk_speed_m_s < speed_table(runner)[FLAT, SurfaceClass.ROCK, 3]
