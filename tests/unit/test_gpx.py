"""GPX export.

Assertions parse the output back with `gpxpy` rather than matching strings:
what matters is that a GPS unit can read it, not how the XML is spelled.
"""

from __future__ import annotations

from pathlib import Path

import gpxpy
import numpy as np
import pytest

from planner_core.core.geo import LatLon
from planner_core.export.gpx import ACCESS_DISCLAIMER, route_to_gpx, write_gpx
from planner_core.graph.spatial_index import SpatialIndex
from planner_core.graph.store import GraphStore
from planner_core.routing.point_to_point import plan_point_to_point
from planner_core.routing.result import Route, build_route
from tests.fixtures.networks import GRID_NW, GRID_SE

pytestmark = pytest.mark.unit


@pytest.fixture
def route(ardennes_graph: GraphStore, ardennes_index: SpatialIndex, profile) -> Route:
    """A real planned route across the fixture grid."""
    plan = plan_point_to_point(
        ardennes_graph,
        ardennes_index,
        LatLon(*GRID_NW),
        LatLon(*GRID_SE),
        max_snap_distance_m=250.0,
        profile=profile,
    )
    return plan.route


def test_output_parses_as_gpx(route: Route):
    parsed = gpxpy.parse(route_to_gpx(route, name="Test ride"))

    assert len(parsed.tracks) == 1
    assert parsed.tracks[0].name == "Test ride"


def test_every_route_point_is_exported(route: Route):
    parsed = gpxpy.parse(route_to_gpx(route, name="Test ride"))

    points = parsed.tracks[0].segments[0].points
    assert len(points) == route.stats.n_points


def test_exported_coordinates_match_the_route(route: Route):
    parsed = gpxpy.parse(route_to_gpx(route, name="Test ride"))

    points = parsed.tracks[0].segments[0].points
    assert points[0].latitude == pytest.approx(route.geometry[0, 0], abs=1e-7)
    assert points[0].longitude == pytest.approx(route.geometry[0, 1], abs=1e-7)
    assert points[-1].latitude == pytest.approx(route.geometry[-1, 0], abs=1e-7)


def test_the_track_is_marked_as_cycling(route: Route):
    parsed = gpxpy.parse(route_to_gpx(route, name="Test ride"))

    assert parsed.tracks[0].type == "cycling"


def test_the_access_disclaimer_travels_with_the_file(route: Route):
    # A GPX file outlives the session that made it, so the caveat that OSM
    # does not model access law has to be inside the file itself.
    xml = route_to_gpx(route, name="Test ride")

    assert ACCESS_DISCLAIMER in xml
    assert "not confirmation that the route is legal" in xml


def test_the_distance_summary_is_included(route: Route):
    xml = route_to_gpx(route, name="Test ride")

    assert f"{route.stats.distance_km:.2f} km" in xml


def test_a_supplied_description_is_preserved(route: Route):
    xml = route_to_gpx(route, name="Test ride", description="Planned from luxembourg")

    assert "Planned from luxembourg" in xml


def test_gpx_length_agrees_with_the_reported_distance(route: Route):
    # Guards against geometry and statistics drifting apart.
    parsed = gpxpy.parse(route_to_gpx(route, name="Test ride"))

    assert parsed.length_2d() == pytest.approx(route.stats.distance_m, rel=0.02)


def test_writing_creates_missing_parent_directories(route: Route, tmp_path: Path):
    target = tmp_path / "nested" / "deeper" / "ride.gpx"

    written = write_gpx(route, target, name="Test ride")

    assert written == target
    assert target.exists()
    assert gpxpy.parse(target.read_text(encoding="utf-8")).tracks


def test_a_single_point_route_still_exports(ardennes_graph: GraphStore):
    degenerate = build_route(ardennes_graph, source=0, arcs=[])

    parsed = gpxpy.parse(route_to_gpx(degenerate, name="Nowhere"))

    assert len(parsed.tracks[0].segments[0].points) == 1


def test_route_geometry_is_two_dimensional(route: Route):
    assert route.geometry.ndim == 2
    assert route.geometry.shape[1] == 2
    assert np.isfinite(route.geometry).all()
