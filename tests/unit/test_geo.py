"""Geometry primitives.

Assertions are anchored to analytically known values rather than to numbers
produced by the implementation itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mtb_planner.core.geo import (
    BBox,
    LatLon,
    bearing_deg,
    densify,
    haversine_m,
    heading_sector,
    polyline_length_m,
)
from mtb_planner.core.units import EARTH_RADIUS_M

pytestmark = pytest.mark.unit

#: One degree of latitude on a sphere of the configured radius.
ONE_DEGREE_M = EARTH_RADIUS_M * math.pi / 180.0


def test_haversine_of_one_degree_latitude_matches_the_analytic_value():
    # Arrange
    a, b = LatLon(0.0, 0.0), LatLon(1.0, 0.0)

    # Act
    distance = haversine_m(a, b)

    # Assert
    assert distance == pytest.approx(ONE_DEGREE_M, rel=1e-9)


def test_haversine_of_one_degree_longitude_at_the_equator_matches_latitude():
    assert haversine_m(LatLon(0.0, 0.0), LatLon(0.0, 1.0)) == pytest.approx(ONE_DEGREE_M, rel=1e-9)


def test_haversine_shrinks_with_latitude_for_the_same_longitude_span():
    at_equator = haversine_m(LatLon(0.0, 0.0), LatLon(0.0, 1.0))
    at_belgium = haversine_m(LatLon(50.85, 4.35), LatLon(50.85, 5.35))

    # cos(50.85 degrees) of the equatorial span.
    assert at_belgium == pytest.approx(at_equator * math.cos(math.radians(50.85)), rel=1e-3)


def test_haversine_of_a_point_with_itself_is_zero():
    assert haversine_m(LatLon(50.13, 5.79), LatLon(50.13, 5.79)) == 0.0


def test_scaled_coordinates_round_trip():
    # Arrange — a coordinate with more precision than 1e-7 degrees preserves.
    original = LatLon(50.1331234, 5.7921567)

    # Act
    restored = LatLon.from_scaled(*original.to_scaled())

    # Assert — 1e-7 degrees is roughly 1 cm.
    assert restored.lat == pytest.approx(original.lat, abs=1e-7)
    assert restored.lon == pytest.approx(original.lon, abs=1e-7)


def test_scaled_coordinates_are_integers():
    lat_e7, lon_e7 = LatLon(50.1331234, 5.7921567).to_scaled()

    assert isinstance(lat_e7, int)
    assert isinstance(lon_e7, int)
    assert lat_e7 == 501331234


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        (LatLon(50.0, 3.0), True),
        (LatLon(49.0, 2.0), True),  # corner, inclusive
        (LatLon(52.0, 7.0), True),  # opposite corner, inclusive
        (LatLon(48.9, 3.0), False),  # south of the box
        (LatLon(50.0, 7.1), False),  # east of the box
    ],
)
def test_bbox_contains(point, expected):
    box = BBox(min_lon=2.0, min_lat=49.0, max_lon=7.0, max_lat=52.0)

    assert box.contains(point) is expected


def test_bbox_expanded_grows_every_side():
    box = BBox(2.0, 49.0, 7.0, 52.0).expanded(0.5)

    assert box == BBox(1.5, 48.5, 7.5, 52.5)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (LatLon(1.0, 0.0), 0.0),  # north
        (LatLon(0.0, 1.0), 90.0),  # east
        (LatLon(-1.0, 0.0), 180.0),  # south
        (LatLon(0.0, -1.0), 270.0),  # west
    ],
)
def test_bearing_from_the_origin_points_the_right_way(target, expected):
    assert bearing_deg(LatLon(0.0, 0.0), target) == pytest.approx(expected, abs=1e-6)


def test_bearing_is_always_in_range():
    assert 0.0 <= bearing_deg(LatLon(50.0, 5.0), LatLon(49.0, 4.0)) < 360.0


@pytest.mark.parametrize(
    ("bearing", "expected"),
    [(0.0, 0), (29.9, 0), (30.0, 1), (180.0, 6), (359.9, 11), (360.0, 0)],
)
def test_heading_sector_buckets_twelve_ways(bearing, expected):
    assert heading_sector(bearing, 12) == expected


def test_heading_sector_rejects_zero_sectors():
    with pytest.raises(ValueError, match="at least 1"):
        heading_sector(90.0, 0)


def test_polyline_length_of_fewer_than_two_points_is_zero():
    assert polyline_length_m(np.array([[50.0, 5.0]])) == 0.0


def test_polyline_length_sums_its_segments():
    # Arrange — two consecutive one-degree steps north.
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    # Act / Assert
    assert polyline_length_m(points) == pytest.approx(2 * ONE_DEGREE_M, rel=1e-6)


def test_densify_respects_the_spacing_limit():
    # Arrange — a single ~111 km segment.
    points = np.array([[0.0, 0.0], [1.0, 0.0]])

    # Act
    dense = densify(points, max_spacing_m=25.0)

    # Assert — every resulting segment is within the limit.
    seg = np.diff(dense[:, 0]) * ONE_DEGREE_M
    assert dense.shape[0] > points.shape[0]
    assert np.all(np.abs(seg) <= 25.0 + 1e-6)


def test_densify_preserves_the_endpoints():
    points = np.array([[50.0, 5.0], [50.01, 5.01]])

    dense = densify(points, max_spacing_m=25.0)

    np.testing.assert_allclose(dense[0], points[0])
    np.testing.assert_allclose(dense[-1], points[-1])


def test_densify_leaves_short_segments_alone():
    # Arrange — already finer than the limit.
    points = np.array([[50.0, 5.0], [50.0001, 5.0]])

    dense = densify(points, max_spacing_m=25.0)

    assert dense.shape[0] == 2


def test_densify_is_a_noop_for_degenerate_input():
    single = np.array([[50.0, 5.0]])

    np.testing.assert_array_equal(densify(single, 25.0), single)
