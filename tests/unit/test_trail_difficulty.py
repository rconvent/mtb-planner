"""Difficulty on foot, on the hiking scale."""

from __future__ import annotations

import pytest

from planner_core.features.enums import TECH_SCALE_MAX, SurfaceClass, TechConfidence
from planner_core.features.surface import classify_surface
from planner_core.ingest.tags import WayTags
from trail.difficulty import (
    MTB_SCALE_TECH,
    UPHILL_DISCOUNT,
    estimate_difficulty,
    parse_sac_scale,
)

pytestmark = pytest.mark.unit


def tags(**kwargs: str) -> WayTags:
    """Build `WayTags` from OSM keys, translating `__` to a colon."""
    return WayTags.from_osm({key.replace("__", ":"): value for key, value in kwargs.items()})


def difficulty_of(way: WayTags):
    return estimate_difficulty(way, classify_surface(way))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("hiking", 1.0),
        ("mountain_hiking", 2.0),
        ("demanding_mountain_hiking", 3.0),
        ("difficult_alpine_hiking", 6.0),
        (" Hiking ", 1.0),
        ("T2", None),
        (None, None),
    ],
)
def test_sac_scale_parses_onto_the_shared_ordinal(value, expected):
    assert parse_sac_scale(value) == expected


def test_sac_scale_is_a_survey():
    result = difficulty_of(tags(highway="path", sac_scale="mountain_hiking"))

    assert result.tech == 2.0
    assert result.confidence is TechConfidence.SURVEYED


def test_sac_scale_wins_over_mtb_scale():
    # A rider's grade is a proxy; a walker's grade is the real thing.
    result = difficulty_of(tags(highway="path", sac_scale="hiking", mtb__scale="3"))

    assert result.tech == 1.0
    assert result.confidence is TechConfidence.SURVEYED


def test_mtb_scale_is_a_compressed_proxy():
    result = difficulty_of(tags(highway="path", mtb__scale="4"))

    assert result.confidence is TechConfidence.PROXY
    # An S4 is hard on a bike and a scramble on foot, not an alpine route.
    assert result.tech == MTB_SCALE_TECH[4] < 4.0


def test_a_faint_path_is_harder_than_a_clear_one():
    clear = difficulty_of(tags(highway="path", surface="ground"))
    faint = difficulty_of(tags(highway="path", surface="ground", trail_visibility="bad"))

    assert faint.tech > clear.tech


def test_tarmac_is_easier_than_rock():
    paved = difficulty_of(tags(highway="path", surface="asphalt"))
    rock = difficulty_of(tags(highway="path", surface="rock"))

    assert paved.tech < rock.tech


def test_a_bare_way_class_is_a_default_not_an_estimate():
    assert difficulty_of(tags(highway="path")).confidence is TechConfidence.DEFAULT
    assert difficulty_of(tags(highway="path", surface="dirt")).confidence is (
        TechConfidence.ESTIMATED
    )


def test_uphill_is_a_little_easier():
    result = difficulty_of(tags(highway="path", sac_scale="demanding_mountain_hiking"))

    assert result.tech_up == pytest.approx(result.tech - UPHILL_DISCOUNT)


def test_estimates_stay_on_the_scale():
    worst = tags(
        highway="steps",
        surface="rock",
        smoothness="impassable",
        trail_visibility="no",
        informal="yes",
        width="0.3",
    )
    result = estimate_difficulty(worst, SurfaceClass.ROCK)

    assert 0.0 <= result.tech <= TECH_SCALE_MAX
    assert 0.0 <= result.tech_up <= result.tech


def test_flow_is_not_a_runners_feature():
    assert difficulty_of(tags(highway="path", mtb__type="flowtrail")).flow == 0.0
