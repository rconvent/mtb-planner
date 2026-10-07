"""The app shell, its assets, and the front-end test suite.

Most of the interface is JavaScript, which pytest cannot execute. What it can
check is everything around it: that the page renders, that every asset it asks
for exists and is served, that the integrity hash in the page describes the
bytes actually on disk, and that the security headers are set.

Two regression tests here earn their keep. MapLibre rejects an `oklch()` paint
value outright — the layer fails and the route silently does not draw — so the
style document is asserted to contain none. And the design rule that colour
lives only in `tokens.css` is checked rather than trusted, because the moment a
second file names a colour the palette has quietly forked.

The pure front-end logic is tested by Node, and this shells out to it so one
command still runs everything.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from planner_core.api.routes.pages import (
    STATIC_ROOT,
    content_security_policy,
    vendor_manifest,
)

pytestmark = pytest.mark.integration

#: Assets the page cannot work without.
REQUIRED_ASSETS = (
    "css/tokens.css",
    "css/app.css",
    "css/fonts.css",
    "js/app.js",
    "js/map.js",
    "js/profile.js",
    "js/ramp.js",
    "js/route.js",
    "js/urlstate.js",
    "js/format.js",
    "js/api.js",
    "style/paper.json",
    "vendor/maplibre-gl.js",
    "vendor/maplibre-gl.css",
    "vendor/fonts/inter-tight-400.woff2",
    "vendor/fonts/plex-mono-500.woff2",
)


# --------------------------------------------------------------- the shell


def test_the_shell_is_served(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_the_shell_renders_without_a_graph(client: TestClient):
    # The page must load while the artifact is still being read, so a rider sees
    # controls rather than an error. It asks the API for state afterwards.
    assert "Trail plate" in client.get("/").text


def test_the_shell_carries_the_mount_points_the_script_needs(client: TestClient):
    body = client.get("/").text

    for anchor in ('id="map"', 'id="profile-chart"', 'id="profiles"', 'id="plan"'):
        assert anchor in body, f"missing {anchor}"


def test_the_shell_loads_its_own_code_and_nothing_else(client: TestClient):
    # Code is the invariant, not bytes in general. A basemap adds image requests
    # to a configured tile host; nothing may add a script or a stylesheet.
    body = client.get("/").text

    assert "/static/js/app.js" in body
    assert "/static/vendor/maplibre-gl.js" in body
    assert "https://" not in body


def test_the_page_states_that_difficulty_is_inferred(client: TestClient):
    # A credibility requirement, not a nicety: at a few per cent survey coverage
    # an interface that implies surveyed difficulty everywhere is dishonest.
    body = client.get("/").text

    assert "inferred" in body
    assert "estimated" in body


def test_the_page_does_not_claim_a_route_is_legal(client: TestClient):
    # Access rules come from OSM and the Wallonian forest code is not modelled
    # at all, so the page has to say so.
    body = client.get("/").text.lower()

    assert "openstreetmap" in body
    assert "check signs" in body


# --------------------------------------------------------------- integrity


def test_the_vendored_bundle_is_pinned_with_an_integrity_hash(client: TestClient):
    match = re.search(r'integrity="(sha384-[^"]+)"', client.get("/").text)

    assert match, "the page loads MapLibre without an integrity hash"


def test_the_integrity_hash_describes_the_bytes_on_disk(client: TestClient):
    # A stale hash means the bundle silently fails to execute, which looks like
    # a broken map rather than a mismatched file.
    match = re.search(r'integrity="(sha384-[^"]+)"', client.get("/").text)
    assert match

    body = (STATIC_ROOT / "vendor" / "maplibre-gl.js").read_bytes()
    expected = "sha384-" + base64.b64encode(hashlib.sha384(body).digest()).decode()

    assert match.group(1) == expected


def test_the_manifest_records_the_pinned_version():
    manifest = vendor_manifest()

    assert manifest["maplibre_version"] != "unknown"
    assert manifest["files"]["maplibre-gl.js"]["bytes"] > 100_000


# --------------------------------------------------------------- assets


@pytest.mark.parametrize("asset", REQUIRED_ASSETS)
def test_every_required_asset_is_served(client: TestClient, asset):
    response = client.get(f"/static/{asset}")

    assert response.status_code == 200
    assert len(response.content) > 0


def test_the_stylesheets_the_page_links_all_exist(client: TestClient):
    for href in re.findall(r'<link[^>]+href="(/static/[^"]+)"', client.get("/").text):
        assert client.get(href).status_code == 200, href


def test_every_module_the_entry_point_imports_resolves(client: TestClient):
    # A bad relative import in an ES module fails at run time in the browser and
    # nowhere else, so the import graph is walked here.
    seen: set[str] = set()
    queue = ["js/app.js"]

    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)

        response = client.get(f"/static/{current}")
        assert response.status_code == 200, current

        folder = Path(current).parent
        for target in re.findall(r'from\s+"(\./[^"]+)"', response.text):
            queue.append((folder / target[2:]).as_posix())

    assert len(seen) >= 7


def test_the_static_mount_cannot_shadow_the_api(client: TestClient):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/profiles").status_code == 200


# --------------------------------------------------------------- security


def test_a_content_security_policy_is_set(client: TestClient):
    policy = client.get("/").headers["content-security-policy"]

    assert policy.startswith("default-src 'self'")
    assert "script-src 'self'" in policy


def test_the_policy_forbids_script_from_anywhere_else(client: TestClient):
    # The one thing this page must never do is run code from a third party.
    policy = client.get("/").headers["content-security-policy"]

    assert "script-src 'self'" in policy
    assert "unsafe-inline" not in policy.split("script-src")[1].split(";")[0]
    assert "unsafe-eval" not in policy


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "strict-origin-when-cross-origin"),
    ],
)
def test_the_cheap_security_headers_are_set(client: TestClient, header, expected):
    assert client.get("/").headers[header] == expected


def test_framing_the_page_is_refused(client: TestClient):
    assert "frame-ancestors 'none'" in client.get("/").headers["content-security-policy"]


# --------------------------------------------------------------- the basemap


def test_the_basemap_is_a_setting_not_a_dependency(monkeypatch):
    """Offline has to stay reachable.

    Every other part of this project answers from a local artifact with nothing
    in the request path. A basemap gives that up in exchange for knowing which
    valley you are looking at, so it is a choice with an off switch rather than
    something wired in.
    """
    from planner_core.api.routes.style import build_style
    from planner_core.config.settings import get_settings

    monkeypatch.setenv("MTB_BASEMAP_URL", "")
    get_settings.cache_clear()
    offline = build_style(get_settings())

    assert offline["sources"] == {}
    assert not any(layer["id"] == "basemap" for layer in offline["layers"])


def test_a_configured_basemap_is_injected_under_everything(monkeypatch):
    from planner_core.api.routes.style import build_style
    from planner_core.config.settings import get_settings

    monkeypatch.setenv("MTB_BASEMAP_URL", "https://example.test/{z}/{x}/{y}.png")
    get_settings.cache_clear()
    style = build_style(get_settings())

    assert style["sources"]["basemap"]["tiles"] == ["https://example.test/{z}/{x}/{y}.png"]
    # First, so the trail network and the route draw over it.
    assert style["layers"][0]["id"] == "basemap"


def test_the_basemap_is_credited(monkeypatch):
    # Attribution is a licence condition for OSM data and a usage condition for
    # most tile servers, so it travels on the source itself.
    from planner_core.api.routes.style import build_style
    from planner_core.config.settings import get_settings

    get_settings.cache_clear()
    style = build_style(get_settings())

    assert "OpenStreetMap" in style["sources"]["basemap"]["attribution"]


def test_the_basemap_is_pushed_back_behind_the_route():
    """The difficulty ramp is the only saturated thing on the map.

    A full-strength raster under a route coloured green-to-red-violet turns the
    plate into noise, and the ramp stops reading as meaning.
    """
    from planner_core.api.routes.style import BASEMAP_PAINT

    assert BASEMAP_PAINT["raster-opacity"] < 1.0
    assert BASEMAP_PAINT["raster-saturation"] < 0.0


def test_the_policy_allows_the_tiles_it_is_configured_to_fetch():
    """Both directives, because MapLibre pulls rasters through the Fetch API.

    `img-src` alone silently blocked every tile: the map came up blank with
    nothing but a console error, which is how this was found.
    """
    policy = content_security_policy("https://tiles.example.test")

    directives = dict(part.split(" ", 1) for part in policy.split("; ") if " " in part)
    assert "https://tiles.example.test" in directives["img-src"]
    assert "https://tiles.example.test" in directives["connect-src"]


def test_no_basemap_means_no_extra_origin():
    policy = content_security_policy("")

    assert "https://" not in policy


def test_scripts_are_never_allowed_from_anywhere_else():
    # The half of the policy that must not move, whatever the basemap is.
    for host in ("", "https://tiles.example.test"):
        assert "script-src 'self';" in content_security_policy(host) + ";"


# --------------------------------------------------------------- design rules


def test_colour_is_declared_only_in_the_token_file():
    # The rule that makes retheming one edit. A colour named anywhere else has
    # forked the palette, whether or not it currently looks the same.
    tokens = STATIC_ROOT / "css" / "tokens.css"

    for sheet in (STATIC_ROOT / "css").glob("*.css"):
        if sheet == tokens:
            continue
        text = sheet.read_text(encoding="utf-8")
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", text), f"hex colour in {sheet.name}"
        assert "oklch(" not in text, f"colour literal in {sheet.name}"


def test_the_map_style_holds_no_colour_maplibre_cannot_parse():
    # MapLibre rejects an oklch paint value and drops the layer, so the route
    # simply does not appear. This is the regression guard for that.
    document = json.loads((STATIC_ROOT / "style" / "paper.json").read_text(encoding="utf-8"))

    assert document["version"] == 8
    assert "oklch" not in json.dumps(document["layers"])


def test_the_difficulty_ramp_covers_the_whole_scale():
    tokens = (STATIC_ROOT / "css" / "tokens.css").read_text(encoding="utf-8")

    for step in range(7):
        assert f"--tech-{step}:" in tokens


def _oklch_luminance(lightness: float, chroma: float, hue: float) -> float:
    """Relative luminance of an oklch colour, for a contrast check.

    OKLab through LMS to linear sRGB, the same conversion the front end uses to
    hand colours to the map, then the WCAG luminance weights.
    """
    radians = math.radians(hue)
    a = chroma * math.cos(radians)
    b = chroma * math.sin(radians)

    long_, medium, short = (
        (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3,
        (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3,
        (lightness - 0.0894841775 * a - 1.291485548 * b) ** 3,
    )
    linear = (
        4.0767416621 * long_ - 3.3077115913 * medium + 0.2309699292 * short,
        -1.2684380046 * long_ + 2.6097574011 * medium - 0.3413193965 * short,
        -0.0041960863 * long_ - 0.7034186147 * medium + 1.707614701 * short,
    )
    red, green, blue = (min(1.0, max(0.0, channel)) for channel in linear)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    """WCAG contrast ratio between two oklch colours."""
    a = _oklch_luminance(*first)
    b = _oklch_luminance(*second)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def _token(text: str, name: str, *, inside: str | None = None) -> tuple[float, float, float]:
    """Read one oklch token out of the stylesheet."""
    scope = text
    if inside is not None:
        start = text.index(inside)
        scope = text[start : text.index("}", text.index("{", start))]
    match = re.search(rf"{re.escape(name)}:\s*oklch\(([\d.]+)%\s+([\d.]+)\s+([\d.]+)\)", scope)
    assert match, f"{name} not found"
    return (float(match[1]) / 100, float(match[2]), float(match[3]))


@pytest.mark.parametrize("ink", ["--ink", "--ink-muted", "--ink-faint"])
def test_every_text_tone_clears_wcag_aa_in_light(ink):
    # --ink-faint carries every label and hint at 11 to 13 px, so it is real
    # reading text and needs 4.5:1. It measured 3.53:1 before this was checked.
    tokens = (STATIC_ROOT / "css" / "tokens.css").read_text(encoding="utf-8")
    ratio = _contrast(_token(tokens, ink), _token(tokens, "--paper-raised"))

    assert ratio >= 4.5, f"{ink} is {ratio:.2f}:1 on --paper-raised"


@pytest.mark.parametrize("ink", ["--ink", "--ink-muted", "--ink-faint"])
def test_every_text_tone_clears_wcag_aa_in_dark(ink):
    tokens = (STATIC_ROOT / "css" / "tokens.css").read_text(encoding="utf-8")
    dark = "prefers-color-scheme: dark"
    ratio = _contrast(
        _token(tokens, ink, inside=dark), _token(tokens, "--paper-raised", inside=dark)
    )

    assert ratio >= 4.5, f"{ink} is {ratio:.2f}:1 on dark --paper-raised"


def test_motion_is_gated_on_the_viewer_preference():
    app_css = (STATIC_ROOT / "css" / "app.css").read_text(encoding="utf-8")

    assert "prefers-reduced-motion" in app_css


# --------------------------------------------------------------- javascript


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_front_end_test_suite_passes():
    """Run the Node test suite, so one command covers the whole project.

    The front end has no bundler and no npm dependencies, so its tests run on
    the runtime built into Node. Failures are surfaced in full rather than
    summarised, because this is the only place they appear.
    """
    root = Path(__file__).resolve().parents[2]
    files = sorted(str(path) for path in (root / "tests" / "js").glob("*.test.mjs"))
    assert files, "no front-end tests found"

    result = subprocess.run(
        ["node", "--test", *files],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
