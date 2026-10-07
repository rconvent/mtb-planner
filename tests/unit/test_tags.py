"""OSM tag capture and parsing.

Cases are drawn from values that genuinely occur in the wild. Every one of
them must resolve to a defined answer — a single malformed tag should never
abort a build spanning a country.
"""

from __future__ import annotations

import pytest

from planner_core.ingest.tags import (
    WayTags,
    first_semicolon_value,
    parse_imba_scale,
    parse_layer,
    parse_mtb_scale,
    parse_width_m,
)

pytestmark = pytest.mark.unit

_THIRD = 1.0 / 3.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Plain grades.
        ("0", 0.0),
        ("3", 3.0),
        ("6", 6.0),
        # The S prefix, upper and lower case.
        ("S2", 2.0),
        ("s2", 2.0),
        # Modifiers shift by a third of a grade.
        ("2+", 2.0 + _THIRD),
        ("1-", 1.0 - _THIRD),
        ("0+", _THIRD),
        # Ranges resolve to their midpoint.
        ("1-2", 1.5),
        ("0-2", 1.0),
        ("S1-S2", 1.5),
        # Surrounding whitespace is not a reason to give up.
        ("  2  ", 2.0),
    ],
)
def test_mtb_scale_parses_real_world_values(value, expected):
    assert parse_mtb_scale(value) == pytest.approx(expected)


def test_a_negative_modifier_on_grade_zero_clamps_rather_than_going_negative():
    # "0-" is easier than S0, but the scale has no room below zero.
    assert parse_mtb_scale("0-") == 0.0


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        "7",  # beyond the scale
        "2+3",  # not a grammar the scale has
        "easy",
        "S",
        "-1",
    ],
)
def test_unparseable_mtb_scale_returns_none_rather_than_guessing(value):
    assert parse_mtb_scale(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0.0), ("2", 2.0), ("4", 4.0), ("imba2", 2.0), ("IMBA3", 3.0)],
)
def test_imba_scale_parses(value, expected):
    assert parse_imba_scale(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, "", "5", "9", "black", "-1"])
def test_imba_scale_rejects_out_of_range_and_prose(value):
    assert parse_imba_scale(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.5", 1.5),
        ("1,5", 1.5),  # comma decimal, common in Belgium
        ("1.5 m", 1.5),
        ("1.5m", 1.5),
        ("2", 2.0),
        ("80 cm", 0.8),
        ("80cm", 0.8),
        ("0.6", 0.6),
    ],
)
def test_width_parses_to_metres(value, expected):
    assert parse_width_m(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "about 1m",  # prose
        "narrow",
        "0",  # a zero-width way is a tagging error
        "-1",
        "500",  # implausible for anything rideable
        "1.5 ft",
    ],
)
def test_implausible_or_prose_widths_are_rejected(value):
    assert parse_width_m(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("gravel;dirt", "gravel"),
        ("Gravel", "gravel"),
        ("  ground ; mud ", "ground"),
        ("dirt", "dirt"),
    ],
)
def test_semicolon_lists_take_the_first_value(value, expected):
    assert first_semicolon_value(value) == expected


@pytest.mark.parametrize("value", [None, "", "   ", ";"])
def test_empty_semicolon_values_become_none(value):
    assert first_semicolon_value(value) is None


@pytest.mark.parametrize(
    ("value", "expected"), [("1", 1), ("-1", -1), ("0", 0), (None, 0), ("abc", 0), ("", 0)]
)
def test_layer_defaults_to_ground_level_on_anything_odd(value, expected):
    assert parse_layer(value) == expected


def test_from_osm_maps_colon_separated_keys_to_fields():
    tags = WayTags.from_osm(
        {
            "highway": "path",
            "mtb:scale": "2",
            "mtb:scale:uphill": "1",
            "mtb:scale:imba": "3",
            "mtb:type": "singletrail",
            "oneway:bicycle": "no",
            "cycleway:left": "opposite_lane",
        }
    )

    assert tags.highway == "path"
    assert tags.mtb_scale == "2"
    assert tags.mtb_scale_uphill == "1"
    assert tags.mtb_scale_imba == "3"
    assert tags.mtb_type == "singletrail"
    assert tags.oneway_bicycle == "no"
    assert tags.cycleway_left == "opposite_lane"


def test_from_osm_drops_keys_outside_the_retained_set():
    # Carrying whole tag dictionaries is what makes a country-scale build
    # run out of memory.
    tags = WayTags.from_osm({"highway": "track", "source": "survey", "note": "muddy"})

    assert tags.highway == "track"
    assert not hasattr(tags, "source")


def test_absent_tags_are_none():
    assert WayTags.from_osm({"highway": "path"}).surface is None


@pytest.mark.parametrize(
    ("tags", "bridge", "tunnel"),
    [
        ({}, False, False),
        ({"bridge": "yes"}, True, False),
        ({"bridge": "viaduct"}, True, False),
        ({"bridge": "no"}, False, False),
        ({"tunnel": "yes"}, False, True),
        ({"tunnel": "culvert"}, False, True),
        ({"tunnel": "no"}, False, False),
    ],
)
def test_bridge_and_tunnel_detection(tags, bridge, tunnel):
    # These decide whether the DEM may be sampled at all: reading a 30 m
    # raster across a viaduct returns the valley floor, not the deck.
    parsed = WayTags.from_osm(tags)

    assert parsed.is_bridge is bridge
    assert parsed.is_tunnel is tunnel
