"""The area-of-interest build configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from planner_core.config.aoi import DEFAULT_CONFIG_PATH, AoiConfig, load_aoi_config
from planner_core.core.errors import BuildError

pytestmark = pytest.mark.unit

_MINIMAL = """
artifact = "test-v1"
regions = ["europe/luxembourg"]
"""

_FULL = """
artifact = "aoi-v1"
contract = false
bbox = [1.9, 48.8, 7.9, 53.8]
regions = ["europe/benelux", "europe/germany/saarland"]
"""


def write(directory: Path, body: str) -> Path:
    path = directory / "aoi.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_minimal_config_loads_with_defaults(tmp_path: Path):
    config = load_aoi_config(write(tmp_path, _MINIMAL))

    assert config.artifact == "test-v1"
    assert config.regions == ("europe/luxembourg",)
    assert config.bbox is None
    assert config.contract is True


def test_a_full_config_loads_every_field(tmp_path: Path):
    config = load_aoi_config(write(tmp_path, _FULL))

    assert config.artifact == "aoi-v1"
    assert config.contract is False
    assert config.regions == ("europe/benelux", "europe/germany/saarland")
    box = config.as_bbox()
    assert box is not None
    assert box.min_lon == pytest.approx(1.9)
    assert box.max_lat == pytest.approx(53.8)


def test_no_bbox_means_no_clipping(tmp_path: Path):
    assert load_aoi_config(write(tmp_path, _MINIMAL)).as_bbox() is None


def test_extract_paths_follow_the_download_layout(tmp_path: Path):
    config = load_aoi_config(write(tmp_path, _FULL))

    paths = config.extract_paths(Path("data/osm"))

    assert [p.name for p in paths] == ["benelux-latest.osm.pbf", "saarland-latest.osm.pbf"]


def test_a_missing_config_is_reported_clearly(tmp_path: Path):
    with pytest.raises(BuildError, match="No area-of-interest config"):
        load_aoi_config(tmp_path / "absent.toml")


def test_malformed_toml_is_reported_clearly(tmp_path: Path):
    with pytest.raises(BuildError, match="not valid TOML"):
        load_aoi_config(write(tmp_path, "artifact = "))


@pytest.mark.parametrize(
    "body",
    [
        'regions = ["europe/luxembourg"]',  # no artifact
        'artifact = "x"',  # no regions
        'artifact = "x"\nregions = []',  # empty regions
        'artifact = ""\nregions = ["europe/luxembourg"]',  # blank artifact
    ],
)
def test_configs_missing_essentials_are_rejected(tmp_path: Path, body):
    with pytest.raises(BuildError, match="not a usable"):
        load_aoi_config(write(tmp_path, body))


@pytest.mark.parametrize(
    "bbox",
    [
        "[7.9, 48.8, 1.9, 53.8]",  # longitudes inverted
        "[1.9, 53.8, 7.9, 48.8]",  # latitudes inverted
        "[1.9, 48.8, 1.9, 53.8]",  # zero width
    ],
)
def test_an_inverted_bbox_is_rejected(tmp_path: Path, bbox):
    body = f'artifact = "x"\nregions = ["europe/luxembourg"]\nbbox = {bbox}'

    with pytest.raises(BuildError, match="not a usable"):
        load_aoi_config(write(tmp_path, body))


# --------------------------------------------------------------- shipped config


def test_the_shipped_config_is_loadable():
    # It is the default for `mtb build`, so a typo in it breaks the command
    # for everyone.
    config = load_aoi_config(DEFAULT_CONFIG_PATH)

    assert config.artifact
    assert config.regions
    assert config.as_bbox() is not None


def test_the_shipped_config_covers_the_low_countries():
    # This asserted `regions[0] == "europe/benelux"`, which Geofabrik does not
    # publish. Two tests agreed with the plan rather than with the download
    # server, so a build of the full area of interest would have contained no
    # Belgium and no Netherlands and nothing would have said so.
    regions = load_aoi_config(DEFAULT_CONFIG_PATH).regions

    for country in ("europe/belgium", "europe/netherlands", "europe/luxembourg"):
        assert country in regions, f"the shipped config omits {country}"


def test_the_shipped_bbox_covers_the_belgian_ardennes():
    box = load_aoi_config(DEFAULT_CONFIG_PATH).as_bbox()

    assert box is not None
    # Houffalize, in the middle of the Ardennes network.
    assert box.min_lat <= 50.13 <= box.max_lat
    assert box.min_lon <= 5.79 <= box.max_lon


def test_regions_are_validated_as_slugs(tmp_path: Path):
    config = AoiConfig(artifact="x", regions=("/bad",))

    with pytest.raises(ValueError, match="Invalid region slug"):
        config.extract_paths(tmp_path)
