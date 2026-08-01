"""Unit conversions."""

from __future__ import annotations

import pytest

from mtb_planner.core.units import (
    DISMOUNT_SPEED_M_S,
    MAX_SPEED_M_S,
    MIN_SPEED_M_S,
    km_h_to_m_s,
    km_to_m,
    m_s_to_km_h,
    m_to_km,
    minutes_to_s,
    s_to_minutes,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("to_base", "from_base", "value"),
    [
        (km_to_m, m_to_km, 42.195),
        (minutes_to_s, s_to_minutes, 90.0),
        (km_h_to_m_s, m_s_to_km_h, 25.0),
    ],
)
def test_conversions_round_trip(to_base, from_base, value):
    assert from_base(to_base(value)) == pytest.approx(value)


def test_known_conversion_anchors():
    assert km_to_m(1) == 1000
    assert minutes_to_s(90) == 5400
    # 36 km/h is exactly 10 m/s.
    assert km_h_to_m_s(36.0) == pytest.approx(10.0)


def test_speed_bounds_are_ordered_and_plausible():
    assert MIN_SPEED_M_S < DISMOUNT_SPEED_M_S < MAX_SPEED_M_S
    # The ceiling should be a fast descent, not a motorway speed.
    assert 30.0 < m_s_to_km_h(MAX_SPEED_M_S) < 60.0
