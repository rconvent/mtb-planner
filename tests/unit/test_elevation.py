"""Sampling elevation and reducing it to climb.

The corrections here are the point of the module. Measured against a real
26 km Luxembourg route, raw sampling reports 661 m of ascent, smoothing brings
it to 591 m, and the dead-band to 531 m — a fifth of the raw figure was sensor
noise rather than terrain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from planner_core.core.errors import DemTileMissingError
from planner_core.core.geo import BBox
from planner_core.elevation.dem_catalog import DemTile, tile_for, tiles_for_bbox
from planner_core.elevation.dem_sampler import DemSampler, require_tiles
from planner_core.elevation.dem_unpublished import (
    load_unpublished,
    record_unpublished,
    registry_path,
)
from planner_core.elevation.profile import (
    DEAD_BAND_M,
    GRADE_BANDS,
    EdgeElevation,
    elevate_edges,
    interpolate_structures,
    profile_statistics,
    smooth,
)

pytestmark = pytest.mark.unit

SPACING_M = 25.0


# --------------------------------------------------------------- catalogue


@pytest.mark.parametrize(
    ("lat", "lon", "expected"),
    [
        (49, 6, "Copernicus_DSM_COG_10_N49_00_E006_00_DEM"),
        (50, 5, "Copernicus_DSM_COG_10_N50_00_E005_00_DEM"),
        (0, 0, "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"),
        (-34, 18, "Copernicus_DSM_COG_10_S34_00_E018_00_DEM"),
        (45, -123, "Copernicus_DSM_COG_10_N45_00_W123_00_DEM"),
    ],
)
def test_tiles_are_named_after_their_south_west_corner(lat, lon, expected):
    assert DemTile(lat=lat, lon=lon).name == expected


def test_a_tile_url_nests_the_file_under_a_directory_of_the_same_name():
    tile = DemTile(lat=49, lon=6)

    assert tile.url.endswith(f"/{tile.name}/{tile.name}.tif")
    assert tile.url.startswith("https://")


@pytest.mark.parametrize(
    ("lat", "lon", "expected"),
    [(49.9, 6.1, (49, 6)), (49.0, 6.0, (49, 6)), (-0.5, -0.5, (-1, -1))],
)
def test_a_coordinate_resolves_to_the_tile_below_and_left_of_it(lat, lon, expected):
    tile = tile_for(lat, lon)

    assert (tile.lat, tile.lon) == expected


def test_a_bounding_box_covers_every_tile_it_touches():
    tiles = tiles_for_bbox(BBox(min_lon=5.7, min_lat=49.4, max_lon=6.6, max_lat=50.2))

    assert {(t.lat, t.lon) for t in tiles} == {(49, 5), (49, 6), (50, 5), (50, 6)}


def test_a_box_inside_one_tile_needs_only_that_tile():
    tiles = tiles_for_bbox(BBox(min_lon=6.1, min_lat=49.6, max_lon=6.2, max_lat=49.7))

    assert len(tiles) == 1


# --------------------------------------------------------------- sampling


def write_tile(directory: Path, tile: DemTile, values: np.ndarray) -> Path:
    """Write a synthetic one-degree tile, north-up, at the tile's corner."""
    height, width = values.shape
    path = directory / tile.filename
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(tile.lon, tile.lat + 1, 1 / width, 1 / height),
        nodata=-32768.0,
    ) as sink:
        sink.write(values.astype("float32"), 1)
    return path


@pytest.fixture
def ramp_tile(tmp_path: Path) -> Path:
    """A tile rising 100 m per row from north to south."""
    values = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0], [200.0, 200.0, 200.0]])
    write_tile(tmp_path, DemTile(lat=49, lon=6), values)
    return tmp_path


def test_sampling_a_pixel_centre_returns_its_value(ramp_tile: Path):
    sampler = DemSampler(ramp_tile)
    # Centre of the top row: half a pixel down from the tile's north edge.
    lat = np.array([49 + 1 - 0.5 / 3])
    lon = np.array([6 + 0.5 / 3])

    assert sampler.sample(lat, lon)[0] == pytest.approx(0.0, abs=1.0)


def test_sampling_between_pixels_interpolates(ramp_tile: Path):
    # Halfway between the first and second rows should read halfway between
    # their values. Nearest-neighbour would return one or the other.
    sampler = DemSampler(ramp_tile)
    lat = np.array([49 + 1 - 1.0 / 3])
    lon = np.array([6 + 0.5 / 3])

    assert sampler.sample(lat, lon)[0] == pytest.approx(50.0, abs=5.0)


def test_a_missing_tile_yields_no_elevation_rather_than_a_wrong_one(tmp_path: Path):
    sampler = DemSampler(tmp_path)

    result = sampler.sample(np.array([49.5]), np.array([6.5]))

    assert np.isnan(result[0])
    assert sampler.missing_tiles


def test_sampling_nothing_is_not_an_error(ramp_tile: Path):
    empty = DemSampler(ramp_tile).sample(np.array([]), np.array([]))

    assert empty.shape == (0,)


def test_required_tiles_are_checked_before_a_build_starts(tmp_path: Path):
    # A gap found mid-build has already cost the ingest, and a graph flat in
    # one region looks perfectly valid from outside.
    with pytest.raises(DemTileMissingError, match="fetch dem"):
        require_tiles(tmp_path, [DemTile(lat=49, lon=6)])


def test_present_tiles_pass_the_check(ramp_tile: Path):
    require_tiles(ramp_tile, [DemTile(lat=49, lon=6)])


# ------------------------------------------------- squares that are all sea


def test_nothing_is_recorded_until_the_archive_refuses_something(tmp_path: Path):
    record_unpublished(tmp_path, [])

    assert load_unpublished(tmp_path) == frozenset()
    assert not registry_path(tmp_path).exists()


def test_recorded_squares_survive_a_reread(tmp_path: Path):
    sea = DemTile(lat=53, lon=2).name

    record_unpublished(tmp_path, [sea])

    assert load_unpublished(tmp_path) == frozenset({sea})


def test_a_later_fetch_does_not_forget_an_earlier_one(tmp_path: Path):
    # Fetching one area must not erase what fetching another already
    # established about the coastline.
    first = DemTile(lat=53, lon=2).name
    second = DemTile(lat=54, lon=3).name

    record_unpublished(tmp_path, [first])
    record_unpublished(tmp_path, [second])

    assert load_unpublished(tmp_path) == frozenset({first, second})


@pytest.mark.parametrize(
    "content",
    ["not json at all", '{"schema": 99, "unpublished": []}', '{"unpublished": ["x"]}'],
)
def test_an_unreadable_note_demands_every_tile_again(tmp_path: Path, content: str):
    # Excusing a gap the note cannot account for is the dangerous direction;
    # going back to insisting on every tile merely fails loudly.
    registry_path(tmp_path).write_text(content, encoding="utf-8")

    assert load_unpublished(tmp_path) == frozenset()


def test_a_square_the_archive_never_published_does_not_block_a_build(tmp_path: Path):
    # GLO-30 is land only. An area of interest reaching open water always
    # wants a few tiles nobody can download, and demanding them makes the
    # build impossible rather than safe.
    sea = DemTile(lat=53, lon=2)
    record_unpublished(tmp_path, [sea.name])

    require_tiles(tmp_path, [sea])


def test_an_unfetched_land_tile_still_blocks_a_build(tmp_path: Path):
    record_unpublished(tmp_path, [DemTile(lat=53, lon=2).name])

    with pytest.raises(DemTileMissingError, match="N49_00_E006"):
        require_tiles(tmp_path, [DemTile(lat=53, lon=2), DemTile(lat=49, lon=6)])


# --------------------------------------------------------------- profile


def test_smoothing_flattens_noise_but_keeps_the_shape():
    noisy = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    smoothed = smooth(noisy, SPACING_M)

    assert smoothed.shape == noisy.shape
    assert smoothed.std() < noisy.std()


def test_a_short_profile_is_left_alone():
    short = np.array([10.0, 20.0])

    np.testing.assert_array_equal(smooth(short, SPACING_M), short)


def test_a_steady_climb_is_measured():
    # Ten samples rising 5 m each, well above the dead-band.
    profile = np.arange(10, dtype=np.float64) * 5.0

    ascent, descent, _, _, _ = profile_statistics(profile, SPACING_M)

    assert ascent > 30.0
    assert descent == pytest.approx(0.0)


def test_a_steady_descent_is_measured_as_descent():
    profile = np.arange(10, dtype=np.float64)[::-1] * 5.0

    ascent, descent, _, _, _ = profile_statistics(profile, SPACING_M)

    assert descent > 30.0
    assert ascent == pytest.approx(0.0)


def test_noise_below_the_dead_band_is_not_climb():
    # Oscillation of a few centimetres, repeated: real rasters do this, and
    # summing it invents a mountain over a few thousand samples.
    wobble = np.tile([0.0, DEAD_BAND_M / 4], 60)

    ascent, descent, _, _, _ = profile_statistics(wobble, SPACING_M)

    assert ascent == pytest.approx(0.0)
    assert descent == pytest.approx(0.0)


def test_a_flat_profile_has_no_climb():
    ascent, descent, high, low, _ = profile_statistics(np.zeros(20), SPACING_M)

    assert (ascent, descent, high, low) == (0.0, 0.0, 0.0, 0.0)


def test_grade_bands_sum_to_the_length_measured():
    profile = np.arange(11, dtype=np.float64) * 3.0

    *_, bands = profile_statistics(profile, SPACING_M)

    assert bands.shape == (GRADE_BANDS,)
    # Ten steps of 25 m between eleven samples.
    assert bands.sum() == pytest.approx(10 * SPACING_M)


def test_a_climb_lands_in_a_positive_grade_band():
    # 5 m over 25 m is a 20% grade, which belongs above the midpoint.
    profile = np.arange(10, dtype=np.float64) * 5.0

    *_, bands = profile_statistics(profile, SPACING_M)

    midpoint = GRADE_BANDS // 2
    assert bands[midpoint + 1 :].sum() > bands[:midpoint].sum()


def test_an_unsampled_profile_reports_flat_rather_than_guessing():
    partial = np.array([100.0, np.nan, 120.0])

    ascent, descent, _, _, bands = profile_statistics(partial, SPACING_M)

    assert (ascent, descent) == (0.0, 0.0)
    assert bands.sum() == pytest.approx(0.0)


# --------------------------------------------------------------- structures


def test_a_bridge_profile_becomes_a_straight_line():
    # Sampled across a viaduct the raster dives into the valley and climbs
    # back out; the deck does neither.
    valley = np.array([100.0, 40.0, 30.0, 40.0, 100.0])
    offsets = np.array([0, 5], dtype=np.int64)

    adjusted = interpolate_structures(valley, offsets, np.array([True]))

    np.testing.assert_allclose(adjusted, np.full(5, 100.0))


def test_a_bridge_keeps_a_genuine_difference_between_its_ends():
    sloping = np.array([100.0, 10.0, 10.0, 130.0])
    offsets = np.array([0, 4], dtype=np.int64)

    adjusted = interpolate_structures(sloping, offsets, np.array([True]))

    assert adjusted[0] == pytest.approx(100.0)
    assert adjusted[-1] == pytest.approx(130.0)
    assert np.all(np.diff(adjusted) > 0)


def test_ordinary_edges_are_not_flattened():
    profile = np.array([100.0, 40.0, 30.0, 40.0, 100.0])
    offsets = np.array([0, 5], dtype=np.int64)

    adjusted = interpolate_structures(profile, offsets, np.array([False]))

    np.testing.assert_array_equal(adjusted, profile)


def test_flattening_a_bridge_removes_its_invented_climb():
    valley = np.array([100.0, 40.0, 30.0, 40.0, 100.0])
    offsets = np.array([0, 5], dtype=np.int64)

    raw = elevate_edges(valley, offsets, SPACING_M)
    bridged = elevate_edges(
        interpolate_structures(valley, offsets, np.array([True])), offsets, SPACING_M
    )

    assert raw.ascent_m[0] > 0.0
    assert bridged.ascent_m[0] == pytest.approx(0.0)


# --------------------------------------------------------------- edge arrays


def test_every_edge_gets_an_entry():
    elevations = np.concatenate([np.arange(6) * 4.0, np.zeros(6)])
    offsets = np.array([0, 6, 12], dtype=np.int64)

    result = elevate_edges(elevations, offsets, SPACING_M)

    assert len(result) == 2
    assert result.grade_hist.shape == (2, GRADE_BANDS)


def test_an_empty_elevation_is_flat_everywhere():
    empty = EdgeElevation.empty(4)

    assert len(empty) == 4
    assert empty.total_ascent_m == pytest.approx(0.0)
    assert empty.grade_hist.shape == (4, GRADE_BANDS)


def test_selecting_a_subset_keeps_the_arrays_aligned():
    elevations = np.concatenate([np.arange(6) * 4.0, np.zeros(6)])
    offsets = np.array([0, 6, 12], dtype=np.int64)
    result = elevate_edges(elevations, offsets, SPACING_M)

    subset = result.select(np.array([1], dtype=np.intp))

    assert len(subset) == 1
    assert subset.ascent_m[0] == pytest.approx(result.ascent_m[1])
