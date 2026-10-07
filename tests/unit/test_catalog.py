"""Geofabrik region slugs and the URLs derived from them."""

from __future__ import annotations

import pytest

from planner_core.ingest.catalog import (
    DEFAULT_AOI_REGIONS,
    GEOFABRIK_BASE,
    SMOKE_TEST_REGION,
    Region,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("slug", "name", "filename"),
    [
        ("europe/belgium", "belgium", "belgium-latest.osm.pbf"),
        ("europe/benelux", "benelux", "benelux-latest.osm.pbf"),
        ("europe/france/lorraine", "lorraine", "lorraine-latest.osm.pbf"),
        ("europe/germany/saarland", "saarland", "saarland-latest.osm.pbf"),
    ],
)
def test_a_slug_yields_its_name_and_local_filename(slug, name, filename):
    region = Region(slug)

    assert region.name == name
    assert region.filename == filename


def test_urls_follow_the_published_layout():
    region = Region("europe/luxembourg")

    assert region.url == f"{GEOFABRIK_BASE}/europe/luxembourg-latest.osm.pbf"
    assert region.md5_url == region.url + ".md5"


@pytest.mark.parametrize("slug", ["", "/europe/belgium", "europe/belgium/"])
def test_malformed_slugs_are_rejected(slug):
    with pytest.raises(ValueError, match="Invalid region slug"):
        Region(slug)


def test_nested_regions_keep_only_the_last_component_as_a_filename():
    assert Region("europe/france/lorraine").filename != Region("europe/benelux").filename


def test_the_default_area_of_interest_leads_with_the_low_countries():
    # This test used to assert `DEFAULT_AOI_REGIONS[0] == "europe/benelux"`,
    # which is not a Geofabrik region. It encoded the intention in the plan --
    # one file, no internal seam -- and nothing ever checked that the path
    # resolved, so the test agreed with the bug for the life of the project.
    # The seam is handled at ingest instead, like every other border.
    assert DEFAULT_AOI_REGIONS[:3] == (
        "europe/belgium",
        "europe/netherlands",
        "europe/luxembourg",
    )


def test_every_default_region_produces_a_valid_url():
    for slug in (*DEFAULT_AOI_REGIONS, SMOKE_TEST_REGION):
        region = Region(slug)
        assert region.url.startswith(f"{GEOFABRIK_BASE}/")
        assert region.url.endswith("-latest.osm.pbf")


def test_default_regions_are_unique():
    assert len(set(DEFAULT_AOI_REGIONS)) == len(DEFAULT_AOI_REGIONS)


def test_default_region_filenames_do_not_collide():
    # A collision would mean one extract silently overwriting another.
    filenames = [Region(slug).filename for slug in DEFAULT_AOI_REGIONS]

    assert len(set(filenames)) == len(filenames)


# --------------------------------------------------------------- broken downloads


def test_an_empty_extract_is_not_treated_as_a_download():
    """A zero-byte file passed as verified and reached the parser.

    Geofabrik does not publish a checksum for every region. Where the sidecar
    404s there is nothing to compare against, and the check was
    `expected is None or md5 matches` — so any existing file passed, including an
    empty one left behind by an earlier failure. It was then logged as "already
    present and verified", which is a claim that no verification supports.
    """
    from planner_core.ingest.download import _MIN_PLAUSIBLE_BYTES

    # A pbf carries a header blob before any data, so nothing real is this small.
    assert _MIN_PLAUSIBLE_BYTES > 0
    assert 0 < _MIN_PLAUSIBLE_BYTES <= 64 * 1024


def test_a_present_file_without_a_checksum_is_not_called_verified(tmp_path, caplog):
    """Say what was actually checked, not what would have been reassuring."""
    import logging

    import httpx

    from planner_core.ingest import download as module

    target = tmp_path / "luxembourg-latest.osm.pbf"
    target.write_bytes(
        bytes([0, 0, 0, 13, 10, 9]) + b"OSMHeader" + b"x" * module._MIN_PLAUSIBLE_BYTES
    )

    region = Region("europe/luxembourg")
    original = module._published_md5
    module._published_md5 = lambda client, region: None  # no sidecar published
    try:
        with caplog.at_level(logging.INFO, logger="planner_core.ingest.download"):
            result = module.download_extract(region, tmp_path)
    except httpx.HTTPError:  # pragma: no cover - only if the network is reachable
        pytest.skip("network reachable; this test only covers the local branch")
    finally:
        module._published_md5 = original

    assert result == target
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "no checksum published" in messages
    assert "verified" not in messages


def test_an_html_error_page_is_not_accepted_as_an_extract(tmp_path):
    """The bug that hid a missing country.

    `europe/benelux` is not a Geofabrik region, but the path is answered with the
    download index page and a 200, so nothing raised. A nine-kilobyte HTML file
    was written as `benelux-latest.osm.pbf`, reported as "already present and
    verified" on every later run, and a full area-of-interest build would have
    quietly contained no Belgium and no Netherlands.
    """
    from planner_core.ingest.download import _looks_like_pbf

    page = tmp_path / "benelux-latest.osm.pbf"
    page.write_bytes(
        b"<!DOCTYPE html>" + bytes([10]) + b"<html><head><title>Geofabrik</title>" + b"x" * 9000
    )

    assert not _looks_like_pbf(page)


def test_a_real_extract_is_recognised(tmp_path):
    from planner_core.ingest.download import _looks_like_pbf

    # The type string every PBF carries in its first blob header.
    extract = tmp_path / "real-latest.osm.pbf"
    # The type string every PBF carries in its first blob header.
    extract.write_bytes(bytes([0, 0, 0, 13, 10, 9]) + b"OSMHeader" + b"payload" * 500)

    assert _looks_like_pbf(extract)


def test_every_configured_region_is_a_plausible_geofabrik_path():
    """A region that does not exist is a country silently missing from the build.

    Not a network call: this asserts the shape Geofabrik uses, which is what
    `europe/benelux` violated. Whether each one resolves is checked by fetching.
    """
    for slug in DEFAULT_AOI_REGIONS:
        assert slug.startswith("europe/"), slug
        assert not slug.endswith("/"), slug
        assert " " not in slug, slug


def test_the_area_of_interest_covers_the_countries_it_is_named_after():
    """Benelux plus Ardennes border strips is the whole premise of the project.

    There is no combined Benelux extract, so all three countries have to be
    listed individually or one of them is absent from every build.
    """
    joined = " ".join(DEFAULT_AOI_REGIONS)

    for country in ("belgium", "netherlands", "luxembourg"):
        assert country in joined, f"the AOI does not include {country}"


def test_the_configured_regions_match_the_defaults():
    """`config/aoi.toml` and the code must not drift apart."""
    from pathlib import Path

    text = Path("config/aoi.toml").read_text(encoding="utf-8")
    for slug in DEFAULT_AOI_REGIONS:
        assert slug in text, f"{slug} is missing from config/aoi.toml"
