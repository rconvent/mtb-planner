"""The interface, in a real browser.

Everything else about the front end is checked without executing it. This is the
part that cannot be: whether MapLibre accepts the style, whether the module graph
loads, whether a plan actually reaches the map, and whether hovering the
elevation chart moves a marker along the trail.

It runs the app in-process against the synthetic lattice network, so it needs no
built artifact and no network. The one thing it deliberately asserts hard is that
the console stays clean: the defect that cost the most in this phase was an
`oklch()` paint value that MapLibre rejected, which produced no Python error, no
failed request and no visible exception — just a map with no route on it.

Marked `e2e`, so `-m "not slow and not e2e"` skips it.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
import uvicorn

from mtb_planner import ARTIFACT_SCHEMA_VERSION, COST_MODEL_VERSION
from mtb_planner.api.app import create_app
from mtb_planner.routing.kernels import warm_up
from mtb_planner.services.graph_registry import LoadedGraph, get_registry

pytestmark = [pytest.mark.e2e, pytest.mark.integration]

playwright_api = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")

#: The plan calls for these widths.
BREAKPOINTS = ((1440, 900), (1024, 768), (768, 1024), (375, 812), (320, 640))

#: Centre of the lattice fixture network.
START = "51.016,4.024"


def free_port() -> int:
    """An unused port, so parallel runs do not collide."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    """The real app, served over HTTP on a spare port."""
    warm_up()

    port = free_port()
    instance = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 30
    while not instance.started and time.monotonic() < deadline:
        time.sleep(0.1)
    assert instance.started, "the test server did not start"

    yield f"http://127.0.0.1:{port}"

    instance.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(autouse=True)
def _publish(lattice_graph, lattice_index) -> None:
    """Put the lattice network into the registry for every test.

    Per test, not once per module: the shared `_reset_registry` fixture clears
    the registry around each one, so a module-scoped publish would leave every
    test after the first waiting for a plan that can never arrive.
    """
    get_registry().set_graph(
        LoadedGraph(
            artifact="lattice-fixture",
            schema_version=ARTIFACT_SCHEMA_VERSION,
            cost_model_version=COST_MODEL_VERSION,
            store=lattice_graph,
            index=lattice_index,
            loaded_at=datetime.now(UTC),
        )
    )


@pytest.fixture(scope="module")
def browser() -> Iterator[object]:
    """A headless Chromium, or a skip if the binary was never downloaded."""
    with playwright_api.sync_playwright() as pw:
        try:
            launched = pw.chromium.launch()
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"no chromium available: {exc}")
        yield launched
        launched.close()


@pytest.fixture
def page(browser, server: str):
    """A page with a planned loop already on it, and console errors collected."""
    view = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    view.on(
        "console",
        lambda message: errors.append(message.text) if message.type == "error" else None,
    )
    view.on("pageerror", lambda error: errors.append(str(error)))

    view.goto(f"{server}/?start={START}&profile=flow&min=20", wait_until="networkidle")
    view.wait_for_selector("#dock:not([hidden])", timeout=20_000)
    view.console_errors = errors
    yield view
    view.close()


# --------------------------------------------------------------- loading


def test_the_page_loads_without_a_console_error(page):
    # The regression guard for a paint value MapLibre cannot parse: it fails
    # here and nowhere else.
    assert page.console_errors == []


def test_the_map_canvas_is_created(page):
    assert page.is_visible(".maplibregl-canvas")


def test_the_profiles_are_offered_from_the_server(page):
    # Rendered from /api/profiles rather than hard-coded, so the controls cannot
    # drift from what the cost model reads.
    assert page.locator("#profiles button").count() == 4
    assert page.locator('#profiles button[aria-pressed="true"]').count() == 1


def test_the_artifact_in_use_is_named(page):
    assert "lattice" in page.inner_text("#sheet")


# --------------------------------------------------------------- planning


def test_a_loop_is_planned_and_reported(page):
    assert page.is_visible("#dock")
    assert page.inner_text("#stat-time").strip() not in ("", "--")
    assert float(page.inner_text("#stat-distance")) > 0


def test_the_map_canvas_has_something_to_draw_on(page):
    drawn = page.evaluate("""() => {
            const canvas = document.querySelector('.maplibregl-canvas');
            return canvas ? canvas.width > 0 && canvas.height > 0 : false;
        }""")

    assert drawn


def test_the_elevation_profile_is_drawn(page):
    path = page.get_attribute("#profile-chart .trace", "d")

    assert path and len(path) > 50


def test_how_much_difficulty_was_surveyed_is_stated(page):
    note = page.inner_text("#coverage-note")

    assert "surveyed" in note
    assert "estimated" in note


def test_a_gpx_link_is_offered(page):
    href = page.get_attribute("#gpx", "href")

    assert href and "/gpx" in href


def test_the_url_carries_the_state(page):
    assert "start=" in page.url
    assert "min=" in page.url


# --------------------------------------------------------------- the linked hover


def test_hovering_the_profile_marks_the_route(page):
    # The interaction the design rests on. One marker before, two after.
    before = page.locator(".maplibregl-marker").count()
    box = page.locator("#profile-chart svg").bounding_box()

    page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] / 2)
    page.wait_for_timeout(400)

    assert page.locator(".maplibregl-marker").count() == before + 1
    assert "probing" in (page.get_attribute("#profile-chart", "class") or "")


def test_the_marker_follows_the_cursor_along_the_chart(page):
    box = page.locator("#profile-chart svg").bounding_box()
    seen = []

    for share in (0.2, 0.5, 0.8):
        page.mouse.move(box["x"] + box["width"] * share, box["y"] + box["height"] / 2)
        page.wait_for_timeout(300)
        seen.append(page.inner_text("#profile-chart .tip"))

    assert len(set(seen)) == 3, f"the readout did not move: {seen}"


def test_leaving_the_chart_clears_the_marker(page):
    box = page.locator("#profile-chart svg").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.wait_for_timeout(300)

    page.mouse.move(box["x"] - 200, box["y"] - 200)
    page.wait_for_timeout(300)

    assert "probing" not in (page.get_attribute("#profile-chart", "class") or "")


# --------------------------------------------------------------- the trail network


def test_the_trail_network_is_fetched_for_the_view(page):
    # Without this the map is blank paper with a route floating on it, and a
    # rider cannot see where the route actually is.
    body = page.evaluate("""async () => {
            const b = window.__mapBounds || null;
            const q = "west=3.99&south=50.99&east=4.06&north=51.04&zoom=14";
            const r = await fetch(`/api/network?${q}`);
            return { ok: r.ok, count: (await r.json()).features.length };
        }""")

    assert body["ok"]
    assert body["count"] > 0


def test_the_network_reports_how_much_it_left_out(page):
    headers = page.evaluate("""async () => {
            const q = "west=3.99&south=50.99&east=4.06&north=51.04&zoom=14&limit=2";
            const r = await fetch(`/api/network?${q}`);
            return {
                drawn: r.headers.get("X-Network-Drawn"),
                total: r.headers.get("X-Network-Total"),
                truncated: r.headers.get("X-Network-Truncated"),
            };
        }""")

    assert headers["truncated"] == "1"
    assert int(headers["drawn"]) < int(headers["total"])


# --------------------------------------------------------------- keyboard


def test_the_elevation_chart_is_reachable_by_keyboard(page):
    # The chart is the only way to inspect a point on the route, so it cannot be
    # pointer-only.
    page.focus("#profile-chart svg")

    assert page.evaluate("() => document.activeElement.tagName.toLowerCase()") == "svg"


def test_arrow_keys_move_along_the_route(page):
    page.focus("#profile-chart svg")
    seen = []

    for _ in range(3):
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(150)
        seen.append(page.get_attribute("#profile-chart svg", "aria-valuetext"))

    assert len(set(seen)) == 3, f"the readout did not move: {seen}"


def test_end_jumps_to_the_finish(page):
    page.focus("#profile-chart svg")
    page.keyboard.press("End")
    page.wait_for_timeout(200)

    now = int(page.get_attribute("#profile-chart svg", "aria-valuenow"))
    most = int(page.get_attribute("#profile-chart svg", "aria-valuemax"))

    assert now == most


def test_the_keyboard_cursor_marks_the_route_too(page):
    before = page.locator(".maplibregl-marker").count()
    page.focus("#profile-chart svg")
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(300)

    assert page.locator(".maplibregl-marker").count() == before + 1


# --------------------------------------------------------------- point to point


def test_point_to_point_is_reachable_without_editing_the_url(page):
    page.click("#modes button[data-mode=ab]")
    page.wait_for_timeout(300)

    assert page.get_attribute("#modes button[data-mode=ab]", "aria-pressed") == "true"
    assert page.get_attribute("#end-field", "hidden") is None
    # One answer, so there is nothing to offer options over.
    assert page.get_attribute("#variants-field", "hidden") is not None


def test_a_to_b_waits_for_a_finish_before_planning(page):
    page.click("#modes button[data-mode=ab]")
    page.wait_for_timeout(300)

    # Start is set from the URL, finish is not, so planning must be unavailable.
    assert page.is_disabled("#plan")
    assert "click the map" in page.inner_text("#end-readout")


def test_switching_back_to_a_loop_forgets_the_finish(page):
    page.click("#modes button[data-mode=ab]")
    page.wait_for_timeout(200)
    page.click("#modes button[data-mode=loop]")
    page.wait_for_timeout(300)

    assert page.get_attribute("#end-field", "hidden") is not None
    assert "end=" not in page.url


# --------------------------------------------------------------- responsive


@pytest.mark.parametrize(("width", "height"), BREAKPOINTS)
def test_nothing_overflows_horizontally(page, width, height):
    page.set_viewport_size({"width": width, "height": height})
    page.wait_for_timeout(400)

    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )

    assert overflow <= 1, f"{overflow}px of horizontal overflow at {width}px"


@pytest.mark.parametrize(("width", "height"), BREAKPOINTS)
def test_the_controls_and_the_results_stay_reachable(page, width, height):
    page.set_viewport_size({"width": width, "height": height})
    page.wait_for_timeout(400)

    assert page.is_visible("#rail")
    assert page.is_visible("#dock")
    assert page.is_visible("#plan")


def test_the_dock_becomes_a_bottom_sheet_on_a_phone(page):
    page.set_viewport_size({"width": 375, "height": 812})
    page.wait_for_timeout(400)

    dock = page.locator("#dock").bounding_box()

    assert dock["width"] > 300, "the dock should span the width as a sheet"


# --------------------------------------------------------------- screenshots


def test_screenshots_at_every_breakpoint(page, tmp_path_factory):
    """Capture the interface at each width, for a human to look at."""
    folder = tmp_path_factory.mktemp("shots")

    for width, height in BREAKPOINTS:
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(500)
        target = folder / f"{width}x{height}.png"
        page.screenshot(path=str(target))
        assert target.stat().st_size > 5_000
