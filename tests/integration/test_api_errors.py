"""Error handling at the HTTP boundary.

Failures must arrive in the same envelope as successes, carry a stable code,
and — where the client can act on it — structured detail. Routes here are
mounted only for the test; they exercise the handlers, not real endpoints.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from planner_core.api.app import create_app
from planner_core.api.schemas.common import LatLonModel
from planner_core.core.errors import (
    GraphNotLoadedError,
    IncompatibleArtifactError,
    NoRouteFoundError,
    PointOffNetworkError,
    PointOutsideCoverageError,
    RouteExpiredError,
    ServerBusyError,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def error_client() -> Iterator[TestClient]:
    """A client whose app raises each error type on demand."""
    app = create_app()

    @app.get("/boom/graph-not-loaded")
    def _graph_not_loaded() -> None:
        raise GraphNotLoadedError("No routing artifact at data/graph/aoi-v1")

    @app.get("/boom/artifact")
    def _artifact() -> None:
        raise IncompatibleArtifactError("schema_version 2 != 1")

    @app.get("/boom/busy")
    def _busy() -> None:
        raise ServerBusyError("Too many concurrent plans")

    @app.get("/boom/off-network")
    def _off_network() -> None:
        raise PointOffNetworkError(
            lat=50.1331,
            lon=5.7921,
            nearest_lat=50.1340,
            nearest_lon=5.7930,
            distance_m=118.4,
        )

    @app.get("/boom/outside-coverage")
    def _outside() -> None:
        raise PointOutsideCoverageError(lat=12.0, lon=34.0)

    @app.get("/boom/no-route")
    def _no_route() -> None:
        raise NoRouteFoundError(reason="budget_too_small", detail="Try at least 30 minutes")

    @app.get("/boom/expired")
    def _expired() -> None:
        raise RouteExpiredError("Route 01K3 is no longer available")

    @app.get("/boom/unexpected")
    def _unexpected() -> None:
        raise RuntimeError("a database password leaked into the message")

    @app.post("/boom/validate")
    def _validate(point: LatLonModel) -> LatLonModel:
        return point

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.mark.parametrize(
    ("path", "expected_status", "expected_code"),
    [
        ("/boom/graph-not-loaded", 503, "graph_not_loaded"),
        ("/boom/artifact", 503, "artifact_unavailable"),
        ("/boom/busy", 503, "server_busy"),
        ("/boom/off-network", 422, "point_off_network"),
        ("/boom/outside-coverage", 422, "point_outside_coverage"),
        ("/boom/no-route", 422, "no_route_found"),
        ("/boom/expired", 410, "route_expired"),
        ("/boom/unexpected", 500, "internal_error"),
    ],
)
def test_each_error_maps_to_its_status_and_code(
    error_client: TestClient, path, expected_status, expected_code
):
    response = error_client.get(path)

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code


def test_failures_use_the_same_envelope_as_successes(error_client: TestClient):
    body = error_client.get("/boom/no-route").json()

    assert set(body) == {"success", "data", "error", "meta"}
    assert body["success"] is False
    assert body["data"] is None


def test_off_network_returns_the_nearest_point_so_the_ui_can_offer_a_snap(
    error_client: TestClient,
):
    detail = error_client.get("/boom/off-network").json()["error"]["detail"]

    assert detail["requested"] == {"lat": 50.1331, "lon": 5.7921}
    assert detail["nearest"] == {"lat": 50.1340, "lon": 5.7930}
    assert detail["distance_m"] == pytest.approx(118.4)


def test_no_route_found_exposes_a_machine_readable_reason(error_client: TestClient):
    error = error_client.get("/boom/no-route").json()["error"]

    assert error["detail"]["reason"] == "budget_too_small"
    assert error["message"] == "Try at least 30 minutes"


def test_errors_without_structured_context_omit_detail(error_client: TestClient):
    assert error_client.get("/boom/graph-not-loaded").json()["error"]["detail"] is None


def test_unexpected_errors_do_not_leak_internals_to_the_client(error_client: TestClient):
    body = error_client.get("/boom/unexpected").json()

    assert body["error"]["message"] == "An unexpected error occurred"
    # Flattened so a leaked substring cannot hide inside nested structures.
    assert "password" not in repr(body)


def test_request_validation_failures_use_the_envelope_too(error_client: TestClient):
    # Latitude 91 is outside the range LatLonModel enforces.
    response = error_client.post("/boom/validate", json={"lat": 91.0, "lon": 5.0})

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["detail"]["errors"]


def test_valid_requests_still_pass_through(error_client: TestClient):
    response = error_client.post("/boom/validate", json={"lat": 50.13, "lon": 5.79})

    assert response.status_code == 200
    assert response.json() == {"lat": 50.13, "lon": 5.79}
