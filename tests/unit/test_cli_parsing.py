"""Command line coordinate parsing.

These are the first thing a user touches, so the failure messages matter as
much as the successes.
"""

from __future__ import annotations

import pytest
import typer

from planner_core.cli.parsing import parse_bbox, parse_latlon

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "lat", "lon"),
    [
        ("50.1331,5.7921", 50.1331, 5.7921),
        ("50.1331, 5.7921", 50.1331, 5.7921),  # a space after the comma
        (" 50.1331 , 5.7921 ", 50.1331, 5.7921),
        ("-33.9,18.4", -33.9, 18.4),  # southern and eastern hemispheres
        ("0,0", 0.0, 0.0),
        ("50,5", 50.0, 5.0),  # integers are coordinates too
    ],
)
def test_coordinates_parse(text, lat, lon):
    point = parse_latlon(text)

    assert point.lat == pytest.approx(lat)
    assert point.lon == pytest.approx(lon)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "50.1331",  # only one number
        "50.1331,5.7921,3",  # three
        "fifty,five",
        "50.1331;5.7921",  # wrong separator
        ",",
    ],
)
def test_malformed_coordinates_are_rejected(text):
    with pytest.raises(typer.BadParameter):
        parse_latlon(text)


@pytest.mark.parametrize("text", ["91,5", "-91,5", "50,181", "50,-181"])
def test_coordinates_outside_the_globe_are_rejected(text):
    with pytest.raises(typer.BadParameter):
        parse_latlon(text)


def test_the_error_names_the_offending_option():
    # "--from must be 'lat,lon'" is actionable; "invalid input" is not.
    with pytest.raises(typer.BadParameter, match="--from"):
        parse_latlon("nonsense", "--from")


def test_no_bbox_means_no_clipping():
    assert parse_bbox(None) is None


def test_a_bbox_parses_in_geojson_order():
    box = parse_bbox("2.0,49.0,7.0,52.0")

    assert box is not None
    assert box.min_lon == pytest.approx(2.0)
    assert box.min_lat == pytest.approx(49.0)
    assert box.max_lon == pytest.approx(7.0)
    assert box.max_lat == pytest.approx(52.0)


@pytest.mark.parametrize(
    "text",
    [
        "2.0,49.0,7.0",  # too few
        "2.0,49.0,7.0,52.0,1.0",  # too many
        "a,b,c,d",
        "7.0,49.0,2.0,52.0",  # longitudes inverted
        "2.0,52.0,7.0,49.0",  # latitudes inverted
        "2.0,49.0,2.0,52.0",  # zero width
    ],
)
def test_malformed_or_inverted_bboxes_are_rejected(text):
    with pytest.raises(typer.BadParameter):
        parse_bbox(text)
