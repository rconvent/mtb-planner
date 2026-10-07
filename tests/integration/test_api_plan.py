"""The planning endpoints over HTTP.

These were the one part of Phase 7 verified only by hand against a running
server. What they need to pin is not the routing — that has its own tests —
but the boundary: the envelope, the status code for each way a request can
fail, and the fact that a planned route survives long enough to be downloaded.

`routing_client` publishes the fixture network into the registry, so a plan
here exercises the real cost model and the real compiled kernels.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from planner_core.api.concurrency import PlanSlots
from planner_core.api.deps import plan_slots, route_cache
from planner_core.config.settings import get_settings
from planner_core.cost.profiles import get_profile
from tests.fixtures.networks import FAR_AWAY, GRID_NW, GRID_SE, LATTICE_NE, LATTICE_SW

pytestmark = pytest.mark.integration

#: A point inside the artifact's bounding box but nowhere near a way: the gap
#: between the rideable grid and the isolated fragment to its north-east.
OFF_NETWORK = (50.150, 5.150)


@pytest.fixture(autouse=True)
def _clear_route_cache() -> Iterator[None]:
    """The cache is process-wide, so a route from one test must not answer another."""
    route_cache().clear()
    yield
    route_cache().clear()


def plan_body(
    start: tuple[float, float] = GRID_NW,
    end: tuple[float, float] = GRID_SE,
    **extra: object,
) -> dict[str, object]:
    """A point-to-point request body."""
    return {
        "start": {"lat": start[0], "lon": start[1]},
        "end": {"lat": end[0], "lon": end[1]},
        **extra,
    }


def plan(client: TestClient, **kwargs: object):
    """POST a plan request."""
    return client.post("/api/routes/point-to-point", json=plan_body(**kwargs))


# --------------------------------------------------------------- profiles


def test_profiles_are_listed_in_the_standard_envelope(routing_client: TestClient):
    response = routing_client.get("/api/profiles")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"success", "data", "error", "meta"}
    assert body["success"] is True


def test_every_shipped_profile_is_offered(routing_client: TestClient):
    data = routing_client.get("/api/profiles").json()["data"]

    assert {p["id"] for p in data["profiles"]} == {"xc", "flow", "tech", "gravity"}


def test_each_profile_describes_itself_to_a_rider(routing_client: TestClient):
    profiles = routing_client.get("/api/profiles").json()["data"]["profiles"]

    for entry in profiles:
        assert entry["name"]
        assert entry["description"]
        assert 0.0 <= entry["tech_target"] <= entry["tech_max"]


def test_the_controls_are_declared_by_the_server(routing_client: TestClient):
    # So a front end renders what the cost model actually reads, rather than
    # hard-coding sliders that drift out of step with it.
    data = routing_client.get("/api/profiles").json()["data"]

    assert {s["id"] for s in data["sliders"]} == {"tech_target", "detour", "effort"}
    for slider in data["sliders"]:
        assert slider["min"] < slider["max"]
        assert slider["step"] > 0
    assert "tech" in data["features"]


def test_profiles_are_available_before_any_graph_is_loaded(client: TestClient):
    # A rider should see the controls while the artifact is still loading.
    assert client.get("/api/profiles").status_code == 200


# --------------------------------------------------------------- planning


def test_a_plan_returns_one_route_in_the_standard_envelope(routing_client: TestClient):
    response = plan(routing_client)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["error"] is None
    assert len(body["data"]["routes"]) == 1


def test_a_planned_route_carries_geometry_a_map_can_draw(routing_client: TestClient):
    route = plan(routing_client).json()["data"]["routes"][0]

    assert route["geometry"]["type"] == "LineString"
    assert len(route["geometry"]["coordinates"]) >= 2
    # Longitude first, and in the fixture network's corner of Belgium.
    longitude, latitude = route["geometry"]["coordinates"][0]
    assert 5.0 < longitude < 6.0
    assert 50.0 < latitude < 51.0


def test_the_reported_figures_are_internally_consistent(routing_client: TestClient):
    stats = plan(routing_client).json()["data"]["routes"][0]["stats"]

    assert stats["distance_m"] > 0
    assert stats["distance_km"] == pytest.approx(stats["distance_m"] / 1000.0, abs=0.01)
    assert stats["moving_time_min"] == pytest.approx(stats["moving_time_s"] / 60.0, abs=0.05)
    assert stats["n_edges"] >= 1


def test_a_route_reports_where_each_end_attached_to_the_network(routing_client: TestClient):
    route = plan(routing_client).json()["data"]["routes"][0]

    for snap in (route["start_snap"], route["end_snap"]):
        assert snap["distance_m"] >= 0.0
        assert 50.0 < snap["point"]["lat"] < 51.0


def test_the_response_says_which_profile_and_artifact_answered(routing_client: TestClient):
    # Without this a reported route cannot be reproduced later.
    meta = plan(routing_client, profile="tech").json()["meta"]

    assert meta["profile"] == "tech"
    assert meta["artifact"] == "ardennes-fixture"
    assert meta["settled"] >= 1


def test_slider_overrides_are_accepted(routing_client: TestClient):
    response = plan(routing_client, profile="tech", tech_target=1.5, detour=0.5, effort=250.0)

    assert response.status_code == 200


def test_the_response_says_how_far_it_was_willing_to_detour(routing_client: TestClient):
    # Part of reproducing a reported route: with a budget the λ is chosen per
    # request, so the profile id alone no longer identifies what was searched.
    meta = plan(routing_client, profile="tech").json()["meta"]

    assert meta["detour"] == pytest.approx(get_profile("tech").detour)


def test_a_time_budget_is_accepted_and_answered(routing_client: TestClient):
    response = plan(routing_client, budget_minutes=120)

    assert response.status_code == 200
    route = response.json()["data"]["routes"][0]
    assert route["within_budget"] is True
    assert route["stats"]["moving_time_s"] <= 120 * 60


def test_a_budget_the_ride_cannot_meet_is_answered_with_a_warning(lattice_client: TestClient):
    # Not an error: the rider still gets a route to where they asked to go, and
    # is told how long it actually takes. The lattice is used because the
    # ardennes fixture is 1.5 km of trail, which fits the smallest budget the
    # API accepts and so cannot exercise this at all.
    response = lattice_client.post(
        "/api/routes/point-to-point",
        json=plan_body(start=LATTICE_SW, end=LATTICE_NE, budget_minutes=15),
    )

    assert response.status_code == 200
    route = response.json()["data"]["routes"][0]
    assert route["within_budget"] is False
    assert any("longer than the 15 you allowed" in note for note in route["warnings"])


def test_more_time_buys_more_of_what_the_profile_wants(lattice_client: TestClient):
    # The complaint this answers: the time control did nothing for a
    # point-to-point route, because the budget was never sent and nothing would
    # have read it. A larger budget must be visible in the answer.
    def detour_at(minutes: int) -> float:
        response = lattice_client.post(
            "/api/routes/point-to-point",
            json=plan_body(start=LATTICE_SW, end=LATTICE_NE, budget_minutes=minutes),
        )
        return response.json()["meta"]["detour"]

    assert detour_at(600) > detour_at(15)


def test_no_budget_leaves_the_verdict_unstated(routing_client: TestClient):
    route = plan(routing_client).json()["data"]["routes"][0]

    assert route["within_budget"] is None


@pytest.mark.parametrize("minutes", [14, 601])
def test_a_budget_outside_the_allowed_range_is_rejected(routing_client: TestClient, minutes: int):
    assert plan(routing_client, budget_minutes=minutes).status_code == 422


def test_the_same_request_plans_the_same_route_twice(routing_client: TestClient):
    first = plan(routing_client).json()["data"]["routes"][0]
    second = plan(routing_client).json()["data"]["routes"][0]

    assert first["geometry"] == second["geometry"]
    assert first["stats"] == second["stats"]
    # A fresh id each time, though — the route is cached per request.
    assert first["route_id"] != second["route_id"]


def test_a_start_dragged_a_long_way_to_a_trail_is_flagged(routing_client: TestClient):
    # 225 m north of the grid: inside the 250 m snap limit, so the plan
    # succeeds, but past half of it — far enough that the rider should be told
    # the pin moved rather than wondering why the route begins elsewhere.
    response = plan(routing_client, start=(50.1045, 5.100))

    route = response.json()["data"]["routes"][0]
    assert any("was moved" in warning for warning in route["warnings"])
    assert route["start_snap"]["distance_m"] == pytest.approx(225.2, abs=1.0)


def test_a_pushing_section_is_called_out_to_the_rider(routing_client: TestClient):
    # The fixture network's steps are rideable only on foot. Whichever route
    # includes them must say so rather than quietly counting it as riding.
    route = plan(routing_client).json()["data"]["routes"][0]

    if route["stats"]["walk_m"] > 0:
        assert any("walked" in warning for warning in route["warnings"])


# --------------------------------------------------------------- download


def test_a_planned_route_can_be_downloaded_as_gpx(routing_client: TestClient):
    route = plan(routing_client).json()["data"]["routes"][0]

    response = routing_client.get(route["gpx_url"])

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/gpx+xml")
    assert "<trkpt" in response.text


def test_the_gpx_url_names_the_route_it_belongs_to(routing_client: TestClient):
    route = plan(routing_client).json()["data"]["routes"][0]

    assert route["gpx_url"] == f"/api/routes/{route['route_id']}/gpx"


def test_the_download_arrives_as_a_named_file(routing_client: TestClient):
    # A plain <a download> link is the whole reason the route cache exists.
    route = plan(routing_client).json()["data"]["routes"][0]

    disposition = routing_client.get(route["gpx_url"]).headers["content-disposition"]

    assert disposition.startswith("attachment;")
    assert disposition.endswith('.gpx"')


def test_the_gpx_track_follows_the_geometry_that_was_returned(routing_client: TestClient):
    route = plan(routing_client).json()["data"]["routes"][0]

    body = routing_client.get(route["gpx_url"]).text

    assert body.count("<trkpt") == len(route["geometry"]["coordinates"])


def test_a_route_can_be_re_read_without_planning_it_again(routing_client: TestClient):
    planned = plan(routing_client).json()["data"]["routes"][0]

    response = routing_client.get(f"/api/routes/{planned['route_id']}")

    assert response.status_code == 200
    reread = response.json()["data"]
    assert reread["route_id"] == planned["route_id"]
    assert reread["stats"] == planned["stats"]
    assert reread["geometry"] == planned["geometry"]


@pytest.mark.parametrize("path", ["/api/routes/0123456789ab", "/api/routes/0123456789ab/gpx"])
def test_a_route_the_server_no_longer_holds_is_gone_not_missing(routing_client: TestClient, path):
    # 410 rather than 404: the id was well-formed, the route has simply
    # expired, and the fix is to plan it again rather than correct the URL.
    response = routing_client.get(path)

    assert response.status_code == 410
    assert response.json()["error"]["code"] == "route_expired"


@pytest.mark.parametrize("route_id", ["not-an-id", "ABCDEF012345", "0123456789abcd", "0123"])
def test_something_that_is_not_a_route_id_is_a_bad_request(routing_client: TestClient, route_id):
    # Claiming a mistyped URL is an expired route sends the rider off to
    # re-plan a route that never existed.
    response = routing_client.get(f"/api/routes/{route_id}")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_the_planning_path_is_not_mistaken_for_a_route_id(routing_client: TestClient):
    # `GET /api/routes/point-to-point` reaches the retrieval endpoint, because
    # a bare `{route_id}` matches anything. It must not answer "that route
    # expired" — and Phase 8's `/api/routes/loop` depends on the same guard.
    response = routing_client.get("/api/routes/point-to-point")

    assert response.json()["error"]["code"] != "route_expired"
    assert response.status_code == 422


# --------------------------------------------------------------- failures


def test_planning_without_a_graph_is_unavailable_not_a_bad_request(client: TestClient):
    response = client.post("/api/routes/point-to-point", json=plan_body())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "graph_not_loaded"


def test_a_point_outside_the_built_area_says_so(routing_client: TestClient):
    response = plan(routing_client, start=FAR_AWAY)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "point_outside_coverage"
    assert body["error"]["detail"]["requested"] == {"lat": FAR_AWAY[0], "lon": FAR_AWAY[1]}


def test_a_point_with_no_trail_near_it_offers_the_nearest_one(routing_client: TestClient):
    response = plan(routing_client, start=OFF_NETWORK)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "point_off_network"
    assert error["detail"]["nearest"]["lat"]
    assert error["detail"]["distance_m"] > 0


def test_planning_from_a_point_to_itself_is_rejected_clearly(routing_client: TestClient):
    response = plan(routing_client, start=GRID_NW, end=GRID_NW)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "no_route_found"
    assert error["detail"]["reason"] == "endpoints_coincide"


def test_an_unknown_profile_is_the_client_s_mistake_not_a_server_fault(
    routing_client: TestClient,
):
    # A typo in a profile id must not read as "the server is broken", and the
    # message has to name the profiles that do exist.
    response = plan(routing_client, profile="downhill")

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "unknown_profile"
    assert "flow" in error["message"]


def test_a_saturated_server_asks_the_client_to_come_back(routing_client: TestClient):
    # Refusing beats queueing: a rider watching a spinner cannot tell "busy"
    # from "broken", and a 503 lets the client say which it is.
    routing_client.app.dependency_overrides[plan_slots] = lambda: PlanSlots(0)
    try:
        response = plan(routing_client)
    finally:
        routing_client.app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "server_busy"


def test_a_refused_plan_does_not_consume_a_slot(routing_client: TestClient):
    # One slot, used twice in sequence: the guard must release on the way out
    # or the server locks itself up after the first request.
    routing_client.app.dependency_overrides[plan_slots] = lambda: PlanSlots(1)
    try:
        first = plan(routing_client)
        second = plan(routing_client)
    finally:
        routing_client.app.dependency_overrides.clear()

    assert first.status_code == 200
    assert second.status_code == 200


def test_a_failed_plan_releases_its_slot_too(routing_client: TestClient):
    # The failure path runs inside the guard, so an exception must not leak a
    # slot — otherwise a handful of off-network clicks disables planning.
    routing_client.app.dependency_overrides[plan_slots] = lambda: PlanSlots(1)
    try:
        assert plan(routing_client, start=OFF_NETWORK).status_code == 422
        recovered = plan(routing_client)
    finally:
        routing_client.app.dependency_overrides.clear()

    assert recovered.status_code == 200


def test_a_request_that_outruns_the_configured_timeout_is_abandoned_honestly(
    routing_client: TestClient, monkeypatch
):
    # Proves MTB_ROUTE_TIMEOUT_S actually reaches the search. Two contortions
    # are needed because the fixture network is seventeen nodes: the ceiling
    # drops to one settled node so the first round cannot finish, and the
    # budget is a nanosecond so it is certainly spent by the time the round
    # ends — otherwise the deepening loop simply succeeds on the second try,
    # which is exactly what it is supposed to do.
    monkeypatch.setenv("MTB_ROUTE_TIMEOUT_S", "0.000000001")
    monkeypatch.setattr("planner_core.routing.router.INITIAL_SETTLE_CEILING", 1)
    get_settings.cache_clear()

    response = plan(routing_client)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "no_route_found"
    # Not "disconnected": these two corners are connected, we just stopped
    # looking, and the rider should be told to retry rather than move the pin.
    assert error["detail"]["reason"] == "search_timeout"


def test_the_default_timeout_is_ample_for_a_normal_plan(routing_client: TestClient):
    assert get_settings().route_timeout_s >= 1.0
    assert plan(routing_client).status_code == 200


def test_a_malformed_request_is_a_validation_failure(routing_client: TestClient):
    response = routing_client.post("/api/routes/point-to-point", json={"start": {"lat": 50.1}})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# --------------------------------------------------------------- documentation


def test_the_planning_endpoints_are_documented(routing_client: TestClient):
    paths = routing_client.get("/openapi.json").json()["paths"]

    assert "/api/routes/point-to-point" in paths
    assert "/api/profiles" in paths
    assert "/api/routes/{route_id}/gpx" in paths
