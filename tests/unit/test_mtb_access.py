"""Bike access and direction rules.

Several of these encode decisions that are easy to get wrong and expensive
when wrong: `access=forestry` is not a bicycle ban, contraflow tagging must be
honoured, and a stray `oneway` on a path must not sever the network.
"""

from __future__ import annotations

import pytest

from mtb_planner.access import resolve_access, resolve_direction
from mtb_planner.sport import MTB
from planner_core.ingest.tags import WayTags
from planner_core.sport import Access, Direction

pytestmark = pytest.mark.unit


def tags(**kwargs: str) -> WayTags:
    """Build `WayTags` from OSM keys, translating `__` to a colon."""
    return WayTags.from_osm({key.replace("__", ":"): value for key, value in kwargs.items()})


# --------------------------------------------------------------- access


@pytest.mark.parametrize(
    ("way", "expected"),
    [
        # Ordinary rideable ways.
        (tags(highway="path"), Access.YES),
        (tags(highway="track"), Access.YES),
        (tags(highway="cycleway"), Access.YES),
        (tags(highway="footway"), Access.YES),
        (tags(highway="residential"), Access.YES),
        (tags(highway="pedestrian"), Access.YES),
        (tags(highway="service"), Access.YES),
        # Bicycles banned outright, or not a way at all.
        (tags(highway="motorway"), Access.NO),
        (tags(highway="trunk"), Access.NO),
        (tags(highway="construction"), Access.NO),
        (tags(highway="platform"), Access.NO),
        (tags(), Access.NO),
        (tags(highway="path", area="yes"), Access.NO),
        (tags(highway="path", smoothness="impassable"), Access.NO),
    ],
)
def test_access_by_way_class(way, expected):
    assert resolve_access(way) is expected


@pytest.mark.parametrize(
    ("way", "expected"),
    [
        # Forestry access restricts motor traffic, not bicycles. Excluding it
        # would delete most of the Wallonian forest network.
        (tags(highway="track", access="forestry"), Access.YES),
        (tags(highway="track", motor_vehicle="forestry"), Access.YES),
        (tags(highway="track", access="agricultural"), Access.YES),
        # Genuine prohibitions.
        (tags(highway="path", bicycle="no"), Access.NO),
        (tags(highway="path", access="private"), Access.NO),
        (tags(highway="path", access="no"), Access.NO),
        # An explicit bicycle permission overrides a closed `access`.
        (tags(highway="path", access="private", bicycle="yes"), Access.YES),
        (tags(highway="path", access="no", bicycle="designated"), Access.YES),
        (tags(highway="path", access="private", bicycle="permissive"), Access.YES),
        # ...but never overrides an explicit bicycle ban.
        (tags(highway="path", access="yes", bicycle="no"), Access.NO),
    ],
)
def test_access_restrictions(way, expected):
    assert resolve_access(way) is expected


@pytest.mark.parametrize(
    ("way", "expected"),
    [
        (tags(highway="service", service="driveway"), Access.NO),
        (tags(highway="service", service="parking_aisle"), Access.NO),
        (tags(highway="service", service="alley"), Access.YES),
        # An explicit bicycle tag rescues it.
        (tags(highway="service", service="driveway", bicycle="designated"), Access.YES),
    ],
)
def test_service_roads_that_only_reach_a_driveway_are_dropped(way, expected):
    assert resolve_access(way) is expected


@pytest.mark.parametrize(
    "way",
    [
        tags(highway="steps"),
        tags(highway="path", bicycle="dismount"),
        tags(highway="footway", bicycle="dismount"),
    ],
)
def test_ways_you_have_to_push_are_kept_as_dismount(way):
    # Excluding these would sever legitimate links; riding them is what is
    # wrong, not using them.
    assert resolve_access(way) is Access.WALK


def test_ferries_are_rideable_despite_having_no_highway_tag():
    assert resolve_access(tags(route="ferry")) is Access.YES


@pytest.mark.parametrize(
    ("way", "expected"),
    [
        (tags(highway="path"), True),
        (tags(highway="steps"), True),
        (tags(highway="path", bicycle="no"), False),
        (tags(highway="motorway"), False),
    ],
)
def test_bike_relevance_admits_everything_except_outright_exclusions(way, expected):
    assert MTB.is_relevant(way) is expected


def test_semicolon_lists_do_not_defeat_the_access_rules():
    assert resolve_access(tags(highway="path", surface="gravel;dirt")) is Access.YES


# --------------------------------------------------------------- direction


@pytest.mark.parametrize(
    ("way", "expected"),
    [
        (tags(), Direction.BOTH),
        (tags(highway="residential"), Direction.BOTH),
        (tags(highway="residential", oneway="no"), Direction.BOTH),
        (tags(highway="residential", oneway="yes"), Direction.FORWARD),
        (tags(highway="residential", oneway="true"), Direction.FORWARD),
        (tags(highway="residential", oneway="-1"), Direction.BACKWARD),
        (tags(highway="residential", oneway="reverse"), Direction.BACKWARD),
        (tags(highway="residential", junction="roundabout"), Direction.FORWARD),
    ],
)
def test_direction_on_roads(way, expected):
    assert resolve_direction(way).direction is expected


@pytest.mark.parametrize(
    ("way", "expected"),
    [
        # Contraflow: one-way for cars, two-way for bikes. 26,594 Belgian ways.
        (tags(highway="residential", oneway="yes", oneway__bicycle="no"), Direction.BOTH),
        (tags(highway="residential", oneway="yes", cycleway="opposite"), Direction.BOTH),
        (tags(highway="residential", oneway="yes", cycleway="opposite_lane"), Direction.BOTH),
        (
            tags(highway="residential", oneway="yes", cycleway__left="opposite_track"),
            Direction.BOTH,
        ),
        # Bicycle-specific tags win over the generic one.
        (tags(highway="residential", oneway="no", oneway__bicycle="yes"), Direction.FORWARD),
        # And mountain-bike-specific wins over bicycle-specific.
        (tags(highway="path", oneway__bicycle="yes", oneway__mtb="-1"), Direction.BACKWARD),
    ],
)
def test_bicycle_specific_direction_tags_take_precedence(way, expected):
    assert resolve_direction(way).direction is expected


def test_oneway_on_a_road_is_trusted():
    rule = resolve_direction(tags(highway="residential", oneway="yes"))

    assert rule.direction is Direction.FORWARD
    assert rule.corroborated is True


def test_bare_oneway_on_a_path_is_treated_as_unreliable():
    # A stray oneway=yes on a path is a common mapping slip, and enforcing it
    # can silently make a whole valley unloopable.
    rule = resolve_direction(tags(highway="path", oneway="yes"))

    assert rule.direction is Direction.FORWARD
    assert rule.corroborated is False


@pytest.mark.parametrize(
    "way",
    [
        tags(highway="path", oneway="yes", mtb__scale="3"),
        tags(highway="path", oneway="yes", mtb__type="downhill"),
        tags(highway="path", oneway="yes", mtb__scale__imba="2"),
        tags(highway="path", oneway="yes", incline="down"),
    ],
)
def test_a_oneway_path_is_trusted_when_something_corroborates_it(way):
    # Descent-only trails are tagged by people who ride them, and those people
    # also record technicality or an explicit downhill incline.
    rule = resolve_direction(way)

    assert rule.direction is Direction.FORWARD
    assert rule.corroborated is True


def test_explicit_two_way_on_a_path_needs_no_corroboration():
    assert resolve_direction(tags(highway="path", oneway="no")).corroborated is True
