"""Application settings, and the example file that documents them.

`.env.example` is where somebody looks to find out what is configurable. It is
prose, so it drifts: a setting added in code and never written down is invisible,
and a key written down that no longer exists is worse — it looks like a working
control and silently does nothing. Both directions are checked here.

The basemap defaults get their own tests because they are the one place this
application reaches outside the machine it runs on, and the wrong default is
either a blank map or an unwanted third-party request.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mtb_planner.config.settings import Settings

pytestmark = pytest.mark.unit

EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"
PREFIX = "MTB_"


def documented_keys() -> set[str]:
    """Every MTB_ setting named in the example file, commented out or not."""
    found = set()
    for line in EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*#?\s*(MTB_[A-Z0-9_]+)\s*=", line)
        if match:
            found.add(match.group(1))
    return found


def field_keys() -> set[str]:
    """Every setting the application actually reads."""
    return {PREFIX + name.upper() for name in Settings.model_fields}


# --------------------------------------------------------------- the example


def test_the_example_documents_something():
    # So the checks below cannot pass by finding nothing.
    assert len(documented_keys()) >= 8


def test_every_documented_key_is_a_real_setting():
    """A key in the example the app does not read is worse than a missing one:
    it looks like a working control and silently does nothing."""
    unknown = documented_keys() - field_keys()

    assert not unknown, f"in .env.example but not in Settings: {sorted(unknown)}"


def test_the_basemap_settings_are_documented():
    """The one place this app talks to a third party should not be something you
    have to read the source to discover."""
    for key in ("MTB_BASEMAP_URL", "MTB_BASEMAP_MAX_ZOOM", "MTB_BASEMAP_ATTRIBUTION"):
        assert key in documented_keys(), f"{key} is not in .env.example"


def test_the_example_says_how_to_turn_the_basemap_off():
    # Offline is the mode every other part of the design assumes.
    text = EXAMPLE.read_text(encoding="utf-8")

    assert "offline" in text.lower()
    assert "MTB_BASEMAP_URL=" in text


def test_the_example_carries_the_tile_usage_policy():
    # Modest personal use is permitted; bulk traffic is not, and somebody
    # pointing this at a shared deployment needs to know before they do.
    assert "operations.osmfoundation.org" in EXAMPLE.read_text(encoding="utf-8")


# --------------------------------------------------------------- basemap


def test_openstreetmap_tiles_are_the_default():
    settings = Settings()

    assert settings.basemap_url == "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    assert settings.basemap_enabled


def test_the_default_template_carries_the_placeholders_maplibre_fills():
    # A URL without them fetches the same tile for the whole world.
    url = Settings().basemap_url

    for placeholder in ("{z}", "{x}", "{y}"):
        assert placeholder in url


def test_an_empty_url_turns_the_basemap_off(monkeypatch):
    monkeypatch.setenv("MTB_BASEMAP_URL", "")

    settings = Settings()

    assert not settings.basemap_enabled
    assert settings.basemap_host == ""


def test_whitespace_is_not_a_basemap(monkeypatch):
    monkeypatch.setenv("MTB_BASEMAP_URL", "   ")

    assert not Settings().basemap_enabled


def test_the_host_is_derived_from_the_url(monkeypatch):
    # Never written down twice: a policy naming a different host from the one
    # actually fetched blocks every tile and leaves a blank map.
    monkeypatch.setenv("MTB_BASEMAP_URL", "https://tiles.example.test/a/{z}/{x}/{y}.png")

    assert Settings().basemap_host == "https://tiles.example.test"


def test_the_default_host_matches_the_default_url():
    assert Settings().basemap_host == "https://tile.openstreetmap.org"


def test_a_url_without_a_scheme_yields_no_host(monkeypatch):
    # Rather than emitting a malformed policy directive.
    monkeypatch.setenv("MTB_BASEMAP_URL", "/local/tiles/{z}/{x}/{y}.png")

    assert Settings().basemap_host == ""


def test_the_basemap_is_credited_by_default():
    assert "OpenStreetMap" in Settings().basemap_attribution


# --------------------------------------------------------------- overrides


def test_the_environment_overrides_the_default(monkeypatch):
    monkeypatch.setenv("MTB_BASEMAP_URL", "https://other.test/{z}/{x}/{y}.png")

    assert Settings().basemap_url == "https://other.test/{z}/{x}/{y}.png"


def test_settings_cannot_be_mutated_after_construction():
    settings = Settings()

    with pytest.raises(Exception, match=r"frozen|immutable|validation"):
        settings.basemap_url = "https://elsewhere.test/{z}/{x}/{y}.png"
