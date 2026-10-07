"""The loop endpoint over HTTP.

Runs against `lattice_client` — about 60 km of trail in a 3.5 km square. The
`ardennes_like` network used everywhere else holds 1.5 km of trail in total,
below the fifteen-minute minimum the schema accepts, so it cannot exercise loop
generation succeeding at all. A couple of tests here deliberately keep using it,
because a sparse network is the only place the failure paths are reachable.

Whether the generator lands three loops inside ±10% on real data is the Phase 8
gate, and lives in `test_loop_gate.py`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from planner_core.api.concurrency import PlanSlots
from planner_core.api.deps import plan_slots, route_cache
from tests.fixtures.networks import FAR_AWAY, GRID_NW, LATTICE_CENTRE

pytestmark = pytest.mark.integration

#: The smallest budget the schema accepts, and comfortably inside what the
#: lattice can provide.
MIN_BUDGET = 15

#: Inside the `ardennes_like` bounding box but far from any way. The lattice has
#: no such gap — every point in it is within about 210 m of a trail — so the
#: off-network path can only be reached on the sparse network.
OFF_NETWORK = (50.150, 5.150)


@pytest.fixture(autouse=True)
def _clear_route_cache() -> Iterator[None]:
    """The cache is process-wide, so one test's loops must not answer another's."""
    route_cache().clear()
    yield
    route_cache().clear()


def loop_body(start: tuple[float, float] = LATTICE_CENTRE, **extra: object) -> dict[str, object]:
    """A loop request body."""
    return {
        "start": {"lat": start[0], "lon": start[1]},
        "budget_minutes": MIN_BUDGET,
        **extra,
    }


def ask(client: TestClient, **kwargs: object):
    """POST a loop request."""
    return client.post("/api/routes/loop", json=loop_body(**kwargs))


# --------------------------------------------------------------- shape


def test_a_loop_request_returns_loops_in_the_standard_envelope(lattice_client: TestClient):
    response = ask(lattice_client)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"success", "data", "error", "meta"}
    assert body["success"] is True
    assert body["data"]["loops"]


def test_it_returns_no_more_loops_than_asked_for(lattice_client: TestClient):
    loops = ask(lattice_client, variants=1).json()["data"]["loops"]

    assert len(loops) == 1


def test_each_loop_carries_geometry_a_map_can_draw(lattice_client: TestClient):
    for entry in ask(lattice_client).json()["data"]["loops"]:
        geometry = entry["geometry"]
        assert geometry["type"] == "LineString"
        assert len(geometry["coordinates"]) >= 3
        # A loop closes, so the first and last points coincide.
        assert geometry["coordinates"][0] == geometry["coordinates"][-1]


def test_the_reported_figures_are_internally_consistent(lattice_client: TestClient):
    for entry in ask(lattice_client).json()["data"]["loops"]:
        stats = entry["stats"]
        assert stats["distance_m"] > 0
        assert stats["distance_km"] == pytest.approx(stats["distance_m"] / 1000.0, abs=0.01)
        assert stats["moving_time_min"] == pytest.approx(stats["moving_time_s"] / 60.0, abs=0.05)


def test_a_loop_climbs_what_it_descends(lattice_client: TestClient):
    # It returns to its start, so the two must balance.
    for entry in ask(lattice_client).json()["data"]["loops"]:
        stats = entry["stats"]
        assert stats["ascent_m"] == pytest.approx(stats["descent_m"], abs=1.0)


def test_overlap_is_reported_so_a_rider_can_judge_it(lattice_client: TestClient):
    for entry in ask(lattice_client).json()["data"]["loops"]:
        assert 0.0 <= entry["overlap_ratio"] <= 0.25


def test_loops_are_offered_best_first(lattice_client: TestClient):
    scores = [entry["score"] for entry in ask(lattice_client, variants=3).json()["data"]["loops"]]

    assert scores == sorted(scores, reverse=True)


def test_the_start_snap_is_reported_once(lattice_client: TestClient):
    # A loop has one endpoint, not two, so it does not carry an end snap.
    entry = ask(lattice_client).json()["data"]["loops"][0]

    assert entry["start_snap"]["distance_m"] >= 0.0
    assert "end_snap" not in entry


# --------------------------------------------------------------- character


def test_each_loop_says_what_kind_of_riding_it_is(lattice_client: TestClient):
    character = ask(lattice_client).json()["data"]["loops"][0]["character"]

    assert set(character) == {
        "technicality",
        "singletrack_fraction",
        "signed_fraction",
        "paved_fraction",
        "traffic",
        "unsurveyed_fraction",
    }


def test_how_much_of_the_difficulty_was_guessed_is_visible(lattice_client: TestClient):
    # At a few percent survey coverage, presenting inferred difficulty as
    # surveyed would be dishonest, so the fraction travels with the figure.
    character = ask(lattice_client).json()["data"]["loops"][0]["character"]

    assert 0.0 <= character["unsurveyed_fraction"] <= 1.0
    assert character["technicality"] >= 0.0


@pytest.mark.parametrize(
    "field",
    ["singletrack_fraction", "signed_fraction", "paved_fraction", "unsurveyed_fraction"],
)
def test_every_character_fraction_is_a_fraction(lattice_client: TestClient, field):
    character = ask(lattice_client).json()["data"]["loops"][0]["character"]

    assert 0.0 <= character[field] <= 1.0


# --------------------------------------------------------------- honesty


def test_the_budget_flag_agrees_with_the_reported_time(lattice_client: TestClient):
    # Either a loop lands inside the tolerance and says so, or it misses and the
    # rider is told in words. What must never happen is a silent miss.
    for entry in ask(lattice_client).json()["data"]["loops"]:
        minutes = entry["stats"]["moving_time_min"]
        inside = abs(minutes - MIN_BUDGET) / MIN_BUDGET <= 0.10

        assert entry["within_budget"] is inside
        if not inside:
            assert any("outside the time asked for" in note for note in entry["warnings"])


def test_loops_land_on_the_requested_budget(lattice_client: TestClient):
    # On a network with room to work, the calibration and repair passes should
    # actually hit the budget rather than merely reporting how far off they are.
    loops = ask(lattice_client, variants=3).json()["data"]["loops"]

    assert any(entry["within_budget"] for entry in loops)


def test_a_loop_that_must_be_walked_says_so(lattice_client: TestClient):
    for entry in ask(lattice_client).json()["data"]["loops"]:
        if entry["stats"]["walk_m"] > 0:
            assert any("walked" in note for note in entry["warnings"])


def test_repeated_ground_is_called_out(lattice_client: TestClient):
    for entry in ask(lattice_client).json()["data"]["loops"]:
        if entry["overlap_ratio"] >= 0.01:
            assert any("ridden twice" in note for note in entry["warnings"])


def test_negligible_overlap_is_not_worth_a_warning(lattice_client: TestClient):
    # Warnings round to the nearest per cent, so a note reading "0% of this loop
    # is ridden twice" is noise that trains a rider to ignore the rest.
    for entry in ask(lattice_client).json()["data"]["loops"]:
        if entry["overlap_ratio"] < 0.01:
            assert not any("ridden twice" in note for note in entry["warnings"])


def test_the_response_says_which_profile_and_artifact_answered(lattice_client: TestClient):
    meta = ask(lattice_client, profile="tech").json()["meta"]

    assert meta["profile"] == "tech"
    assert meta["artifact"] == "lattice-fixture"
    assert meta["settled"] >= 1
    assert meta["truncated"] is False


def test_a_seed_is_echoed_even_though_generation_is_deterministic(lattice_client: TestClient):
    # Echoed so a reported route can be quoted back exactly; the schema says
    # plainly that it changes nothing.
    assert ask(lattice_client, seed=42).json()["meta"]["seed"] == 42


def test_the_same_request_returns_the_same_loops(lattice_client: TestClient):
    first = ask(lattice_client).json()["data"]["loops"]
    second = ask(lattice_client).json()["data"]["loops"]

    assert [e["geometry"] for e in first] == [e["geometry"] for e in second]
    assert [e["stats"] for e in first] == [e["stats"] for e in second]


def test_a_seed_does_not_change_the_result(lattice_client: TestClient):
    without = ask(lattice_client).json()["data"]["loops"][0]["geometry"]
    with_seed = ask(lattice_client, seed=7).json()["data"]["loops"][0]["geometry"]

    assert without == with_seed


# --------------------------------------------------------------- download


def test_each_loop_can_be_downloaded_as_gpx(lattice_client: TestClient):
    # Loops share the point-to-point route cache, so the GPX endpoint serves
    # them without knowing which endpoint produced them.
    entry = ask(lattice_client).json()["data"]["loops"][0]

    response = lattice_client.get(entry["gpx_url"])

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/gpx+xml")
    assert response.text.count("<trkpt") == len(entry["geometry"]["coordinates"])


def test_every_loop_gets_its_own_id(lattice_client: TestClient):
    loops = ask(lattice_client, variants=3).json()["data"]["loops"]

    ids = [entry["route_id"] for entry in loops]
    assert len(set(ids)) == len(ids)


def test_each_loop_can_be_re_read_without_regenerating(lattice_client: TestClient):
    entry = ask(lattice_client).json()["data"]["loops"][0]

    response = lattice_client.get(f"/api/routes/{entry['route_id']}")

    assert response.status_code == 200
    assert response.json()["data"]["geometry"] == entry["geometry"]


# --------------------------------------------------------------- failures


def test_the_loop_path_is_not_mistaken_for_a_route_id(lattice_client: TestClient):
    # The reason route ids are pattern-validated: `{route_id}` matches anything,
    # so without it this would report a route that never existed as expired.
    response = lattice_client.get("/api/routes/loop")

    assert response.json()["error"]["code"] != "route_expired"
    assert response.status_code == 422


def test_generating_without_a_graph_is_unavailable(client: TestClient):
    response = client.post("/api/routes/loop", json=loop_body())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "graph_not_loaded"


def test_a_start_outside_the_built_area_says_so(lattice_client: TestClient):
    response = ask(lattice_client, start=FAR_AWAY)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "point_outside_coverage"


def test_a_start_with_no_trail_near_it_offers_the_nearest(routing_client: TestClient):
    # The sparse network, because the lattice has no gap wide enough.
    response = ask(routing_client, start=OFF_NETWORK)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "point_off_network"
    assert error["detail"]["distance_m"] > 0


def test_a_budget_the_area_cannot_fill_is_reported_clearly(routing_client: TestClient):
    # Fifteen minutes on 1.5 km of trail. Not "no route found" — no *loop*
    # found, which tells the rider it is the area rather than the pin.
    response = ask(routing_client, start=GRID_NW)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "no_route_found"
    assert error["detail"]["reason"] == "no_loop_found"


def test_an_unknown_profile_is_the_clients_mistake(lattice_client: TestClient):
    response = ask(lattice_client, profile="downhill")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_profile"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("budget_minutes", 0),
        ("budget_minutes", 14),
        ("budget_minutes", 601),
        ("variants", 0),
        ("variants", 7),
        ("tech_target", 6.5),
        ("detour", -1.0),
        ("effort", 20.0),
    ],
)
def test_out_of_range_requests_are_rejected_at_the_boundary(
    lattice_client: TestClient, field, value
):
    response = lattice_client.post("/api/routes/loop", json={**loop_body(), field: value})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_a_request_without_a_start_is_rejected(lattice_client: TestClient):
    response = lattice_client.post("/api/routes/loop", json={"budget_minutes": 60})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_a_saturated_server_asks_the_client_to_come_back(lattice_client: TestClient):
    lattice_client.app.dependency_overrides[plan_slots] = lambda: PlanSlots(0)
    try:
        response = ask(lattice_client)
    finally:
        lattice_client.app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "server_busy"


def test_a_failed_request_releases_its_slot(lattice_client: TestClient):
    # A handful of misplaced clicks must not disable loop planning.
    lattice_client.app.dependency_overrides[plan_slots] = lambda: PlanSlots(1)
    try:
        assert ask(lattice_client, start=FAR_AWAY).status_code == 422
        recovered = ask(lattice_client)
    finally:
        lattice_client.app.dependency_overrides.clear()

    assert recovered.status_code == 200


# --------------------------------------------------------------- documentation


def test_the_loop_endpoint_is_documented(lattice_client: TestClient):
    paths = lattice_client.get("/openapi.json").json()["paths"]

    assert "/api/routes/loop" in paths
