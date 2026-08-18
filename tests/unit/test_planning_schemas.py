"""The wire format for planning requests and responses.

Two things are worth pinning here. Requests must reject out-of-range values at
the boundary, because nothing downstream re-checks them. And responses must
convert geometry to GeoJSON's longitude-first order — getting that backwards
puts every route in the wrong hemisphere, which is obvious on a map and
invisible in a unit test that does not look for it.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from mtb_planner.api.schemas.planning import (
    MAX_POWER_W,
    MIN_POWER_W,
    PlanRequest,
    ProfileList,
    ProfileSummary,
    geometry_of,
    stats_of,
)
from mtb_planner.cost.profiles import FEATURE_NAMES, RideProfile
from mtb_planner.routing.result import Route, RouteStats

pytestmark = pytest.mark.unit


def a_route(**overrides: float) -> Route:
    """A route with plausible measurements, overridable per test."""
    stats = {
        "distance_m": 26_432.678,
        "n_edges": 41,
        "n_points": 812,
        "dismount_m": 118.44,
        "ascent_m": 531.44,
        "descent_m": 529.61,
        "moving_time_s": 5_412.9,
    }
    stats.update(overrides)
    return Route(
        arcs=(0,),
        nodes=(0, 1),
        geometry=np.array([[50.1234567, 5.7654321], [50.2, 5.8]]),
        stats=RouteStats(**stats),  # type: ignore[arg-type]
    )


# --------------------------------------------------------------- geometry


def test_geometry_is_longitude_first_as_geojson_requires():
    geometry = geometry_of(a_route())

    assert geometry.type == "LineString"
    assert geometry.coordinates[0] == (5.765432, 50.123457)


def test_coordinates_are_rounded_to_six_decimals():
    # Beyond six decimals is under a tenth of a metre: payload weight for
    # noise the DEM and the OSM source cannot support anyway.
    longitude, latitude = geometry_of(a_route()).coordinates[0]

    assert longitude == 5.765432
    assert latitude == 50.123457


def test_every_geometry_point_survives_the_conversion():
    route = a_route()

    assert len(geometry_of(route).coordinates) == route.geometry.shape[0]


# --------------------------------------------------------------- stats


def test_stats_carry_both_the_raw_and_the_readable_figure():
    stats = stats_of(a_route())

    assert stats.distance_m == 26_432.7
    assert stats.distance_km == 26.43
    assert stats.moving_time_s == 5_412.9
    assert stats.moving_time_min == 90.2


def test_ascent_and_descent_are_reported_separately():
    # A loop's ascent and descent are close but not equal, and collapsing them
    # into one number hides which way a point-to-point route leans.
    stats = stats_of(a_route())

    assert stats.ascent_m == 531.4
    assert stats.descent_m == 529.6


def test_average_speed_is_derived_not_invented():
    stats = stats_of(a_route(distance_m=20_000.0, moving_time_s=3_600.0))

    assert stats.average_speed_km_h == 20.0


def test_a_route_with_no_time_estimate_reports_no_speed_rather_than_dividing_by_zero():
    stats = stats_of(a_route(moving_time_s=0.0))

    assert stats.average_speed_km_h == 0.0


def test_pushing_distance_is_surfaced():
    assert stats_of(a_route()).dismount_m == 118.4


# --------------------------------------------------------------- requests


def test_a_minimal_request_defaults_to_a_profile_and_no_overrides():
    request = PlanRequest.model_validate(
        {"start": {"lat": 50.10, "lon": 5.10}, "end": {"lat": 50.20, "lon": 5.20}}
    )

    assert request.profile == "flow"
    assert request.tech_target is None
    assert request.detour is None
    assert request.power_w is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tech_target", -0.1),
        ("tech_target", 6.1),
        ("detour", -0.1),
        ("detour", 3.1),
        ("power_w", MIN_POWER_W - 1),
        ("power_w", MAX_POWER_W + 1),
        ("profile", ""),
    ],
)
def test_out_of_range_overrides_are_rejected_at_the_boundary(field, value):
    with pytest.raises(ValidationError):
        PlanRequest.model_validate(
            {
                "start": {"lat": 50.10, "lon": 5.10},
                "end": {"lat": 50.20, "lon": 5.20},
                field: value,
            }
        )


@pytest.mark.parametrize("missing", ["start", "end"])
def test_both_endpoints_are_required(missing):
    payload = {"start": {"lat": 50.10, "lon": 5.10}, "end": {"lat": 50.20, "lon": 5.20}}
    del payload[missing]

    with pytest.raises(ValidationError):
        PlanRequest.model_validate(payload)


def test_an_impossible_coordinate_is_rejected():
    with pytest.raises(ValidationError):
        PlanRequest.model_validate(
            {"start": {"lat": 91.0, "lon": 5.10}, "end": {"lat": 50.20, "lon": 5.20}}
        )


def test_a_request_cannot_be_mutated_after_validation():
    request = PlanRequest.model_validate(
        {"start": {"lat": 50.10, "lon": 5.10}, "end": {"lat": 50.20, "lon": 5.20}}
    )

    with pytest.raises(ValidationError):
        request.profile = "tech"


# --------------------------------------------------------------- profiles


def test_a_profile_summary_exposes_what_a_client_needs_to_render_it(profile: RideProfile):
    summary = ProfileSummary.of(profile)

    assert summary.id == profile.id
    assert summary.name == profile.name
    assert summary.tech_target == profile.tech_target
    assert summary.tech_max == profile.tech_max
    assert summary.detour == profile.detour


def test_a_summary_omits_the_weight_vector():
    # Weights are an implementation detail of the cost function; publishing
    # them would invite a client to reason about numbers it cannot interpret.
    assert "weights" not in ProfileSummary.model_fields


def test_the_feature_list_comes_from_the_cost_model_itself(profile: RideProfile):
    # Hard-coding it here is how a slider ends up controlling nothing.
    listing = ProfileList(profiles=(ProfileSummary.of(profile),), sliders=())

    assert listing.features == FEATURE_NAMES
