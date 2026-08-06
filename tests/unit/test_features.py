"""Deriving difficulty, surface and way class from tags.

The difficulty chain is the product: explicit `mtb:scale` covers only a few
percent of the network, so most of these cases exercise inference rather than
the surveyed path.
"""

from __future__ import annotations

import pytest

from mtb_planner.features.classify import classify_way, singletrack_score, traffic_exposure
from mtb_planner.features.difficulty import UPHILL_DISCOUNT, Difficulty, estimate_difficulty
from mtb_planner.features.enums import SurfaceClass, TechConfidence, WayClass
from mtb_planner.features.surface import classify_surface, rollability, surface_is_explicit
from mtb_planner.ingest.tags import WayTags

pytestmark = pytest.mark.unit


def tags(**kwargs: str) -> WayTags:
    """Build `WayTags` from OSM keys, translating `__` to a colon."""
    return WayTags.from_osm({key.replace("__", ":"): value for key, value in kwargs.items()})


def difficulty_of(**kwargs: str) -> Difficulty:
    """Estimate difficulty straight from OSM keys."""
    parsed = tags(**kwargs)
    return estimate_difficulty(parsed, classify_surface(parsed))


# --------------------------------------------------------------- surface


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("asphalt", SurfaceClass.PAVED),
        ("paving_stones", SurfaceClass.PAVED),
        ("compacted", SurfaceClass.COMPACTED),
        ("fine_gravel", SurfaceClass.COMPACTED),
        ("gravel", SurfaceClass.GRAVEL),
        ("dirt", SurfaceClass.GROUND),
        ("ground", SurfaceClass.GROUND),
        ("grass", SurfaceClass.GRASS),
        ("rock", SurfaceClass.ROCK),
        ("cobblestone", SurfaceClass.ROCK),
        ("mud", SurfaceClass.SAND_MUD),
        ("sand", SurfaceClass.SAND_MUD),
    ],
)
def test_surface_values_map_to_classes(surface, expected):
    assert classify_surface(tags(highway="path", surface=surface)) is expected


@pytest.mark.parametrize(
    ("highway", "expected"),
    [
        ("path", SurfaceClass.GROUND),
        ("track", SurfaceClass.COMPACTED),
        ("cycleway", SurfaceClass.PAVED),
        ("residential", SurfaceClass.PAVED),
    ],
)
def test_an_absent_surface_falls_back_to_the_way_class(highway, expected):
    # "Unknown" would propagate as a neutral value and flatten the difference
    # between a forest track and a residential street.
    assert classify_surface(tags(highway=highway)) is expected


def test_an_unrecognised_surface_value_falls_back_too():
    assert classify_surface(tags(highway="path", surface="unobtainium")) is SurfaceClass.GROUND


def test_semicolon_surface_lists_take_the_first_value():
    assert classify_surface(tags(highway="path", surface="gravel;dirt")) is SurfaceClass.GRAVEL


def test_explicit_surface_is_distinguishable_from_a_default():
    assert surface_is_explicit(tags(highway="path", surface="gravel")) is True
    assert surface_is_explicit(tags(highway="path")) is False


def test_smoothness_overrides_surface_for_rollability():
    # A rider's direct judgement beats the material.
    rough_asphalt = tags(highway="residential", surface="asphalt", smoothness="very_bad")

    assert rollability(rough_asphalt, SurfaceClass.PAVED) < 0.5


def test_rollability_falls_back_to_the_surface():
    assert rollability(tags(highway="residential"), SurfaceClass.PAVED) == pytest.approx(1.0)


# --------------------------------------------------------------- difficulty


def test_an_explicit_mtb_scale_is_taken_as_surveyed():
    result = difficulty_of(highway="path", mtb__scale="3")

    assert result.tech == pytest.approx(3.0)
    assert result.confidence is TechConfidence.SURVEYED


def test_a_surveyed_scale_beats_every_other_signal():
    # Even where surface and smoothness would suggest something easier.
    result = difficulty_of(
        highway="path", mtb__scale="4", surface="asphalt", smoothness="excellent"
    )

    assert result.tech == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("imba", "expected"),
    [("0", 0.5), ("1", 1.0), ("2", 2.0), ("3", 3.0), ("4", 4.5)],
)
def test_imba_grades_convert_to_a_proxy(imba, expected):
    result = difficulty_of(highway="path", mtb__scale__imba=imba)

    assert result.tech == pytest.approx(expected)
    assert result.confidence is TechConfidence.PROXY


def test_an_imba_grade_implies_purpose_built_flow():
    assert difficulty_of(highway="path", mtb__scale__imba="2").flow > 0.5


@pytest.mark.parametrize(
    ("sac", "expected"),
    [("hiking", 0.7), ("mountain_hiking", 2.0), ("alpine_hiking", 4.5)],
)
def test_the_hiking_scale_converts_to_a_proxy(sac, expected):
    result = difficulty_of(highway="path", sac_scale=sac)

    assert result.tech == pytest.approx(expected)
    assert result.confidence is TechConfidence.PROXY


def test_surface_evidence_yields_an_estimate():
    result = difficulty_of(highway="path", surface="ground")

    assert result.confidence is TechConfidence.ESTIMATED
    assert result.tech > 0.0


def test_nothing_but_a_way_class_is_only_a_default():
    # The honest floor: this is a guess from the highway tag alone.
    assert difficulty_of(highway="path").confidence is TechConfidence.DEFAULT


def test_a_rough_surface_is_harder_than_a_smooth_one():
    rough = difficulty_of(highway="path", surface="rock")
    smooth = difficulty_of(highway="path", surface="compacted")

    assert rough.tech > smooth.tech


def test_a_rough_tracktype_is_harder_than_a_firm_one():
    assert (
        difficulty_of(highway="track", tracktype="grade5").tech
        > difficulty_of(highway="track", tracktype="grade1").tech
    )


def test_tracktype_only_applies_to_tracks():
    # It has no defined meaning on a path, and using it there would double
    # count against surface.
    with_grade = difficulty_of(highway="path", surface="ground", tracktype="grade5")
    without = difficulty_of(highway="path", surface="ground")

    assert with_grade.tech == pytest.approx(without.tech)


def test_a_narrow_way_is_harder_than_a_wide_one():
    assert (
        difficulty_of(highway="path", surface="ground", width="0.5").tech
        > difficulty_of(highway="path", surface="ground", width="3.0").tech
    )


def test_surface_and_smoothness_are_combined_rather_than_summed():
    # Both describe the same property; adding them would double count.
    both = difficulty_of(highway="path", surface="rock", smoothness="very_bad")

    assert both.tech <= 6.0


def test_technicality_never_leaves_the_scale():
    worst = difficulty_of(
        highway="steps", surface="mud", smoothness="very_horrible", width="0.3", informal="yes"
    )

    assert 0.0 <= worst.tech <= 6.0


def test_an_explicit_uphill_scale_is_used_as_given():
    result = difficulty_of(highway="path", mtb__scale="3", mtb__scale__uphill="1")

    assert result.tech_up == pytest.approx(1.0)


def test_uphill_difficulty_is_discounted_when_untagged():
    # A rooty descent is awkward to climb but rarely as hard.
    result = difficulty_of(highway="path", mtb__scale="3")

    assert result.tech_up == pytest.approx(3.0 - UPHILL_DISCOUNT)


def test_a_flow_trail_is_recognised_from_its_type():
    assert difficulty_of(highway="path", mtb__type="flowtrail").flow == pytest.approx(1.0)


def test_an_ordinary_path_has_no_flow_signal():
    assert difficulty_of(highway="path", surface="ground").flow == pytest.approx(0.0)


# --------------------------------------------------------------- way class


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"highway": "path"}, WayClass.SINGLETRACK),
        ({"highway": "bridleway"}, WayClass.SINGLETRACK),
        ({"highway": "steps"}, WayClass.STEPS),
        ({"highway": "track", "tracktype": "grade1"}, WayClass.FIREROAD),
        ({"highway": "track", "tracktype": "grade5"}, WayClass.DOUBLETRACK),
        ({"highway": "residential"}, WayClass.URBAN),
        ({"highway": "primary"}, WayClass.PAVED_MAJOR),
        ({"highway": "tertiary"}, WayClass.PAVED_MINOR),
        ({"route": "ferry"}, WayClass.FERRY),
    ],
)
def test_way_classification(kwargs, expected):
    parsed = tags(**kwargs)

    assert classify_way(parsed, classify_surface(parsed)) is expected


def test_an_unsealed_cycleway_is_not_treated_as_tarmac():
    parsed = tags(highway="cycleway", surface="gravel")

    assert classify_way(parsed, classify_surface(parsed)) is WayClass.GRAVEL_ROAD


def test_a_very_wide_path_is_not_singletrack():
    parsed = tags(highway="path", width="4.0")

    assert classify_way(parsed, classify_surface(parsed)) is WayClass.DOUBLETRACK


# --------------------------------------------------------------- singletrack


def test_a_narrow_way_is_certainly_singletrack():
    parsed = tags(highway="path", width="0.8")

    assert singletrack_score(parsed, classify_surface(parsed)) == pytest.approx(1.0)


def test_an_untagged_forest_path_is_probably_singletrack():
    parsed = tags(highway="path", surface="ground")

    assert singletrack_score(parsed, classify_surface(parsed)) > 0.8


def test_a_surfaced_path_is_less_likely_to_be_singletrack():
    # Often a walking route through a park rather than a trail.
    surfaced = tags(highway="path", surface="compacted")
    bare = tags(highway="path", surface="ground")

    assert singletrack_score(surfaced, classify_surface(surfaced)) < singletrack_score(
        bare, classify_surface(bare)
    )


def test_a_fire_road_is_not_singletrack():
    parsed = tags(highway="track", tracktype="grade1")

    assert singletrack_score(parsed, classify_surface(parsed)) == pytest.approx(0.0)


def test_a_rough_track_retains_some_trail_character():
    parsed = tags(highway="track", tracktype="grade5")

    assert 0.0 < singletrack_score(parsed, classify_surface(parsed)) < 0.5


def test_roads_are_never_singletrack():
    parsed = tags(highway="secondary")

    assert singletrack_score(parsed, classify_surface(parsed)) == pytest.approx(0.0)


# --------------------------------------------------------------- traffic


@pytest.mark.parametrize(
    ("highway", "lower", "upper"),
    [
        ("path", 0.0, 0.01),
        ("track", 0.0, 0.1),
        ("residential", 0.2, 0.5),
        ("secondary", 0.7, 0.9),
        ("primary", 0.95, 1.0),
    ],
)
def test_traffic_exposure_rises_with_road_class(highway, lower, upper):
    assert lower <= traffic_exposure(tags(highway=highway)) <= upper
