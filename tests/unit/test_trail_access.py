"""Foot access and direction rules.

The cases that matter are the ones where a runner and a rider part ways: a
footpath closed to bikes, a cycleway closed to walkers, a one-way descent that
binds only wheels.
"""

from __future__ import annotations

import pytest

from planner_core.ingest.tags import WayTags
from planner_core.sport import Access, Direction
from trail.access import resolve_access, resolve_direction
from trail.sport import TRAIL

pytestmark = pytest.mark.unit


def tags(**kwargs: str) -> WayTags:
    """Build `WayTags` from OSM keys, translating `__` to a colon."""
    return WayTags.from_osm({key.replace("__", ":"): value for key, value in kwargs.items()})


@pytest.mark.parametrize(
    "way",
    [
        tags(highway="path"),
        tags(highway="footway"),
        tags(highway="track", access="forestry"),
        tags(highway="steps"),
        tags(highway="bridleway"),
        tags(highway="cycleway"),
        tags(highway="residential"),
        # A rider's prohibition, not a runner's.
        tags(highway="path", bicycle="no"),
        # Impassable to wheels is not impassable to feet.
        tags(highway="path", smoothness="impassable"),
        # A private way that names pedestrians is open to them.
        tags(highway="track", access="private", foot="yes"),
    ],
)
def test_ways_open_on_foot(way: WayTags):
    assert resolve_access(way) is Access.YES


@pytest.mark.parametrize(
    "way",
    [
        tags(highway="path", foot="no"),
        tags(highway="cycleway", foot="no"),
        tags(highway="secondary", foot="use_sidepath"),
        tags(highway="track", access="private"),
        tags(highway="service", service="driveway"),
        tags(highway="motorway"),
        tags(highway="trunk"),
        tags(highway="construction"),
        tags(highway="pedestrian", area="yes"),
        tags(natural="cliff"),
    ],
)
def test_ways_closed_on_foot(way: WayTags):
    assert resolve_access(way) is Access.NO
    assert not TRAIL.is_relevant(way)


def test_a_scramble_is_kept_but_walked():
    way = tags(highway="path", sac_scale="demanding_alpine_hiking")

    assert resolve_access(way) is Access.WALK


def test_a_ferry_carries_foot_passengers():
    assert resolve_access(tags(route="ferry")) is Access.YES


def test_a_one_way_street_is_two_way_on_foot():
    assert resolve_direction(tags(highway="residential", oneway="yes")).direction is Direction.BOTH


def test_a_one_way_bike_descent_is_two_way_on_foot():
    way = tags(highway="path", oneway="yes", mtb__scale="3")

    assert resolve_direction(way).direction is Direction.BOTH


@pytest.mark.parametrize(
    ("value", "expected"),
    [("yes", Direction.FORWARD), ("-1", Direction.BACKWARD), ("no", Direction.BOTH)],
)
def test_oneway_foot_is_the_only_direction_a_runner_obeys(value: str, expected: Direction):
    rule = resolve_direction(tags(highway="path", oneway__foot=value))

    assert rule.direction is expected
    assert rule.corroborated
