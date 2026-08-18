"""The health endpoint and the response envelope.

Health must answer even when routing is unavailable — a 503 elsewhere is only
diagnosable if this endpoint still explains why.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mtb_planner import __version__

pytestmark = pytest.mark.integration


def test_health_responds_in_the_standard_envelope(client: TestClient):
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"success", "data", "error", "meta"}
    assert body["success"] is True
    assert body["error"] is None


def test_health_reports_degraded_while_no_graph_is_loaded(client: TestClient):
    # No artifact has been built, so routing is genuinely unavailable and the
    # server should say so rather than claiming readiness.
    data = client.get("/api/health").json()["data"]

    assert data["status"] == "degraded"
    assert data["graph"]["loaded"] is False
    assert data["graph"]["error"]


def test_health_reports_the_application_version(client: TestClient):
    assert client.get("/api/health").json()["data"]["version"] == __version__


def test_uptime_is_non_negative(client: TestClient):
    assert client.get("/api/health").json()["data"]["uptime_s"] >= 0


def test_unknown_paths_are_not_silently_successful(client: TestClient):
    assert client.get("/api/does-not-exist").status_code == 404


def test_openapi_schema_is_served(client: TestClient):
    schema = client.get("/openapi.json").json()

    assert "/api/health" in schema["paths"]
