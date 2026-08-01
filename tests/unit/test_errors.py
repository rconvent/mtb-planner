"""The error vocabulary.

These carry structured context the API turns into actionable responses, so
the payload matters as much as the type.
"""

from __future__ import annotations

import pytest

from mtb_planner.core.errors import (
    ArtifactError,
    BuildError,
    IncompatibleArtifactError,
    MtbPlannerError,
    NoRouteFoundError,
    PlanningError,
    PointOffNetworkError,
    PointOutsideCoverageError,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("subclass", "base"),
    [
        (BuildError, MtbPlannerError),
        (ArtifactError, MtbPlannerError),
        (IncompatibleArtifactError, ArtifactError),
        (PlanningError, MtbPlannerError),
        (PointOffNetworkError, PlanningError),
        (NoRouteFoundError, PlanningError),
    ],
)
def test_error_hierarchy(subclass, base):
    assert issubclass(subclass, base)


def test_point_off_network_keeps_the_nearest_point_for_snapping():
    # Arrange / Act
    exc = PointOffNetworkError(
        lat=50.1331,
        lon=5.7921,
        nearest_lat=50.1340,
        nearest_lon=5.7930,
        distance_m=118.4,
    )

    # Assert
    assert exc.nearest_lat == 50.1340
    assert exc.distance_m == pytest.approx(118.4)
    assert "118 m away" in str(exc)


def test_point_off_network_reads_sensibly_without_a_nearest_point():
    exc = PointOffNetworkError(lat=50.1331, lon=5.7921)

    assert "50.13310" in str(exc)
    assert "away" not in str(exc)


def test_point_outside_coverage_names_the_point():
    exc = PointOutsideCoverageError(lat=12.0, lon=34.0)

    assert exc.lat == 12.0
    assert "outside the routable area" in str(exc)


def test_no_route_found_carries_a_machine_readable_reason():
    exc = NoRouteFoundError(reason="budget_too_small", detail="Try at least 30 minutes")

    assert exc.reason == "budget_too_small"
    assert str(exc) == "Try at least 30 minutes"


def test_no_route_found_falls_back_to_the_reason_as_its_message():
    assert str(NoRouteFoundError(reason="disconnected")) == "disconnected"
