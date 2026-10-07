"""The whole Phase 1 chain, driven through the CLI.

This is the test that proves the point of the phase: a `.osm.pbf` goes in and
a GPX a rider can open comes out.
"""

from __future__ import annotations

import json
from pathlib import Path

import gpxpy
import pytest
from typer.testing import CliRunner

from mtb_planner.cli import app
from tests.fixtures.networks import GRID_NW, GRID_SE, single_track

pytestmark = pytest.mark.integration

runner = CliRunner()


def coord(point: tuple[float, float]) -> str:
    """Format a fixture coordinate the way the CLI expects it."""
    return f"{point[0]},{point[1]}"


def test_route_command_turns_an_extract_into_a_readable_gpx(ardennes_pbf: Path, tmp_path: Path):
    out = tmp_path / "ride.gpx"

    result = runner.invoke(
        app,
        [
            "route",
            "--pbf",
            str(ardennes_pbf),
            "--from",
            coord(GRID_NW),
            "--to",
            coord(GRID_SE),
            "--out",
            str(out),
            "--name",
            "Fixture ride",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()

    parsed = gpxpy.parse(out.read_text(encoding="utf-8"))
    assert parsed.tracks[0].name == "Fixture ride"
    assert len(parsed.tracks[0].segments[0].points) >= 2
    assert parsed.length_2d() == pytest.approx(365.0, abs=5.0)


def test_route_command_reports_what_it_built(ardennes_pbf: Path, tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "route",
            "--pbf",
            str(ardennes_pbf),
            "--from",
            coord(GRID_NW),
            "--to",
            coord(GRID_SE),
            "--out",
            str(tmp_path / "ride.gpx"),
        ],
    )

    assert "distance" in result.output
    assert "0.37 km" in result.output
    # The access caveat belongs on screen as well as in the file.
    assert "access law" in result.output


def test_route_command_rejects_a_malformed_coordinate(ardennes_pbf: Path, tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "route",
            "--pbf",
            str(ardennes_pbf),
            "--from",
            "not-a-coordinate",
            "--to",
            coord(GRID_SE),
            "--out",
            str(tmp_path / "ride.gpx"),
        ],
    )

    assert result.exit_code != 0


def test_route_command_fails_cleanly_when_a_point_is_off_network(
    ardennes_pbf: Path, tmp_path: Path
):
    out = tmp_path / "ride.gpx"

    result = runner.invoke(
        app,
        [
            "route",
            "--pbf",
            str(ardennes_pbf),
            "--from",
            "50.5,5.5",
            "--to",
            coord(GRID_SE),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 1
    assert not out.exists()


def test_inspect_reports_tag_coverage(ardennes_pbf: Path):
    result = runner.invoke(app, ["inspect", str(ardennes_pbf)])

    assert result.exit_code == 0, result.output
    assert "ways scanned" in result.output
    assert "mtb:scale on trails" in result.output
    assert "route=mtb relations" in result.output


def test_inspect_warns_when_difficulty_will_mostly_be_inferred(tmp_path: Path):
    # A network with no mtb:scale at all: the operator must be told the
    # difficulty model is inferring rather than reading surveyed values.
    untagged = single_track(tmp_path)

    result = runner.invoke(app, ["inspect", str(untagged)])

    assert result.exit_code == 0, result.output
    # Collapse whitespace: the console wraps, and a line break must not be
    # the difference between this passing and failing.
    assert "has to be inferred" in " ".join(result.output.split())


def test_inspect_json_output_is_machine_readable(ardennes_pbf: Path):
    result = runner.invoke(app, ["inspect", str(ardennes_pbf), "--json"])

    assert result.exit_code == 0, result.output
    # Logs go to stderr so a caller piping stdout gets clean JSON.
    report = json.loads(result.stdout)

    assert report["total_ways"] == 11
    assert report["route_relations"]["mtb"] == 1
    assert report["route_relations"]["bicycle"] == 1
    assert 0.0 <= report["surveyed_coverage"] <= 1.0


def test_inspect_reports_a_missing_file_rather_than_crashing(tmp_path: Path):
    result = runner.invoke(app, ["inspect", str(tmp_path / "absent.osm.pbf")])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_the_command_surface_is_registered():
    result = runner.invoke(app, ["--help"])

    for command in ("fetch", "inspect", "route", "serve"):
        assert command in result.output
