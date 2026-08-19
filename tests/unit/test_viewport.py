"""The trail network as map data.

This is what makes the map show anything other than the planned route, so the
properties that matter are about honesty and cost: what it selects must be inside
the view, what it leaves out must be reported rather than silently dropped, and it
must not ship a megabyte of sub-pixel stubs to draw a country.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtb_planner.core.geo import BBox
from mtb_planner.features.enums import WayClass
from mtb_planner.graph.store import GraphStore
from mtb_planner.graph.viewport import (
    DEFAULT_LIMIT,
    MIN_PIXELS,
    as_geojson,
    min_length_m,
    network_view,
    select_edges,
    stride_for,
    tile_zoom,
    visible_classes,
)

pytestmark = pytest.mark.unit


def whole_of(store: GraphStore) -> BBox:
    """A box comfortably around a fixture network."""
    box = store.bbox()
    return BBox(
        min_lat=box.min_lat - 0.01,
        min_lon=box.min_lon - 0.01,
        max_lat=box.max_lat + 0.01,
        max_lon=box.max_lon + 0.01,
    )


# --------------------------------------------------------------- zoom rules


def test_trails_are_drawn_at_every_zoom():
    # They are the subject of the map, not scenery.
    for cls in (WayClass.SINGLETRACK, WayClass.DOUBLETRACK, WayClass.FIREROAD):
        assert int(cls) in visible_classes(0)


def test_roads_arrive_only_once_they_help_orient():
    assert int(WayClass.PAVED_MAJOR) not in visible_classes(8)
    assert int(WayClass.PAVED_MAJOR) in visible_classes(12)


def test_urban_clutter_arrives_last():
    assert int(WayClass.URBAN) not in visible_classes(12)
    assert int(WayClass.URBAN) in visible_classes(15)


def test_more_zoom_never_shows_fewer_classes():
    for zoom in range(20):
        assert set(visible_classes(zoom)) <= set(visible_classes(zoom + 1))


def test_geometry_is_decimated_hardest_when_zoomed_out():
    assert stride_for(9) > stride_for(12) > stride_for(16)


def test_every_vertex_is_kept_when_zoomed_right_in():
    assert stride_for(16) == 1


# --------------------------------------------------------------- pixel floor


def test_the_length_floor_shrinks_as_you_zoom_in():
    assert min_length_m(11, 50.0) > min_length_m(15, 50.0)


def test_the_floor_is_a_few_pixels_of_ground():
    # Web Mercator: about 12 m per pixel at zoom 13 and 50 degrees north.
    metres = min_length_m(13, 50.0)

    assert MIN_PIXELS * 5 < metres < MIN_PIXELS * 30


def test_the_floor_accounts_for_latitude():
    # A pixel covers less ground further from the equator.
    assert min_length_m(13, 60.0) < min_length_m(13, 0.0)


# --------------------------------------------------------------- selection


def test_the_whole_fixture_network_is_selected_from_a_generous_box(
    lattice_graph: GraphStore,
):
    edges = select_edges(lattice_graph, whole_of(lattice_graph), zoom=16)

    assert edges.shape[0] > 0
    assert edges.shape[0] <= lattice_graph.n_edges


def test_nothing_is_selected_from_a_box_somewhere_else(lattice_graph: GraphStore):
    elsewhere = BBox(min_lat=10.0, min_lon=10.0, max_lat=11.0, max_lon=11.0)

    assert select_edges(lattice_graph, elsewhere, zoom=16).shape[0] == 0


def test_every_selected_edge_touches_the_view(lattice_graph: GraphStore):
    box = lattice_graph.bbox()
    half = BBox(
        min_lat=box.min_lat,
        min_lon=box.min_lon,
        max_lat=(box.min_lat + box.max_lat) / 2,
        max_lon=box.max_lon,
    )

    for edge in select_edges(lattice_graph, half, zoom=16):
        nodes = (int(lattice_graph.edge_u[edge]), int(lattice_graph.edge_v[edge]))
        assert any(
            half.min_lat <= float(lattice_graph.node_lat[node]) <= half.max_lat
            and half.min_lon <= float(lattice_graph.node_lon[node]) <= half.max_lon
            for node in nodes
        )


def test_selection_is_longest_first(lattice_graph: GraphStore):
    # So truncating drops the least visible edges rather than a corner of the map.
    edges = select_edges(lattice_graph, whole_of(lattice_graph), zoom=16)
    lengths = lattice_graph.edge_length_m[edges]

    assert np.all(np.diff(lengths) <= 1e-9)


def test_zooming_out_selects_no_more_than_zooming_in(lattice_graph: GraphStore):
    box = whole_of(lattice_graph)

    assert select_edges(lattice_graph, box, zoom=10).shape[0] <= (
        select_edges(lattice_graph, box, zoom=16).shape[0]
    )


# --------------------------------------------------------------- rendering


def test_the_view_is_a_geojson_feature_collection(lattice_graph: GraphStore):
    view = network_view(lattice_graph, whole_of(lattice_graph), zoom=16)

    assert view.geojson["type"] == "FeatureCollection"
    assert view.geojson["features"]


def test_every_feature_is_a_drawable_line(lattice_graph: GraphStore):
    view = network_view(lattice_graph, whole_of(lattice_graph), zoom=16)

    for feature in view.geojson["features"]:
        assert feature["geometry"]["type"] == "LineString"
        assert len(feature["geometry"]["coordinates"]) >= 2


def test_coordinates_are_longitude_first(lattice_graph: GraphStore):
    # GeoJSON order. Getting it backwards puts the whole map in the wrong place.
    view = network_view(lattice_graph, whole_of(lattice_graph), zoom=16)
    longitude, latitude = view.geojson["features"][0]["geometry"]["coordinates"][0]

    assert 3.0 < longitude < 5.0
    assert 50.0 < latitude < 52.0


def test_every_feature_says_what_kind_of_way_it_is(lattice_graph: GraphStore):
    # The map styles by class: trails properly, roads as hairlines.
    view = network_view(lattice_graph, whole_of(lattice_graph), zoom=16)

    for feature in view.geojson["features"]:
        assert 0 <= feature["properties"]["class"] <= int(WayClass.UNKNOWN)
        assert isinstance(feature["properties"]["signed"], bool)


def test_a_coarser_zoom_ships_fewer_points(lattice_graph: GraphStore):
    box = whole_of(lattice_graph)

    def vertices(zoom: int) -> int:
        view = network_view(lattice_graph, box, zoom=zoom)
        return sum(len(f["geometry"]["coordinates"]) for f in view.geojson["features"])

    assert vertices(9) < vertices(16)


def test_decimation_keeps_the_far_end_of_every_line(
    ardennes_graph_uncontracted: GraphStore,
):
    # A trail decimated to stop short of its junction reads as a dead end.
    box = whole_of(ardennes_graph_uncontracted)
    coarse = network_view(ardennes_graph_uncontracted, box, zoom=9)
    fine = network_view(ardennes_graph_uncontracted, box, zoom=16)

    ends_coarse = {tuple(f["geometry"]["coordinates"][-1]) for f in coarse.geojson["features"]}
    ends_fine = {tuple(f["geometry"]["coordinates"][-1]) for f in fine.geojson["features"]}

    assert ends_coarse <= ends_fine


# --------------------------------------------------------------- honesty


def test_a_complete_view_is_not_reported_as_truncated(lattice_graph: GraphStore):
    view = network_view(lattice_graph, whole_of(lattice_graph), zoom=16)

    assert not view.truncated
    assert view.drawn == view.total


def test_a_thinned_view_says_so(lattice_graph: GraphStore):
    # The important one. A map showing a fraction of the trails with no notice
    # reads as an area with very few trails.
    view = network_view(lattice_graph, whole_of(lattice_graph), zoom=16, limit=3)

    assert view.drawn == 3
    assert view.total > 3
    assert view.truncated


def test_the_limit_is_respected(lattice_graph: GraphStore):
    view = network_view(lattice_graph, whole_of(lattice_graph), zoom=16, limit=5)

    assert len(view.geojson["features"]) <= 5


def test_the_default_limit_is_something_a_browser_can_draw():
    assert 1_000 <= DEFAULT_LIMIT <= 10_000


def test_an_empty_selection_renders_an_empty_collection(lattice_graph: GraphStore):
    view = as_geojson(lattice_graph, np.array([], dtype=np.intp), zoom=16)

    assert view.geojson["features"] == []
    assert not view.truncated


# --------------------------------------------------------------- zoom guess


def test_a_narrow_span_reads_as_a_closer_zoom():
    wide = BBox(min_lat=49.0, min_lon=5.0, max_lat=50.0, max_lon=7.0)
    tight = BBox(min_lat=49.6, min_lon=6.10, max_lat=49.65, max_lon=6.15)

    assert tile_zoom(tight) > tile_zoom(wide)


def test_the_guessed_zoom_stays_in_range():
    for span in (1e-6, 0.001, 1.0, 180.0, 360.0):
        box = BBox(min_lat=0.0, min_lon=0.0, max_lat=span, max_lon=span)
        assert 0 <= tile_zoom(box) <= 20
