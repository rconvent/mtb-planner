"""Fetching elevation tiles, and telling sea apart from a bad connection.

The archive answers 404 for squares that are entirely water. That is the
normal shape of a rectangular area of interest and gets written down, so later
builds stop asking. A timeout or a 500 is the opposite: the tile exists and
has to be fetched again. Recording one as the other would write real ground
off as ocean and quietly flatten it in every build that followed.
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from planner_core.cli import fetch as fetch_cmd
from planner_core.core.errors import DemTileUnpublishedError, ExtractDownloadError
from planner_core.elevation.dem_catalog import DemTile
from planner_core.elevation.dem_unpublished import load_unpublished, registry_path

pytestmark = pytest.mark.unit

runner = CliRunner()

#: A box small enough to want exactly one tile: N50 E005, the Ardennes.
_ONE_TILE_BBOX = "5.2,50.2,5.4,50.4"
_ONE_TILE = DemTile(lat=50, lon=5)


class _Refusing:
    """Stands in for `httpx.stream`, refusing with a given status."""

    def __init__(self, status: int) -> None:
        self._status = status

    def __call__(self, *args: object, **kwargs: object) -> _Refusing:
        return self

    def __enter__(self) -> _Refusing:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        request = httpx.Request("GET", "https://example.invalid/tile.tif")
        response = httpx.Response(self._status, request=request)
        raise httpx.HTTPStatusError("refused", request=request, response=response)


def test_a_404_means_the_square_was_never_published(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(fetch_cmd.httpx, "stream", _Refusing(HTTPStatus.NOT_FOUND))

    with pytest.raises(DemTileUnpublishedError):
        fetch_cmd._download_tile("https://example.invalid/t.tif", tmp_path / "t.tif")


@pytest.mark.parametrize("status", [HTTPStatus.INTERNAL_SERVER_ERROR, HTTPStatus.FORBIDDEN])
def test_any_other_refusal_is_a_download_failure(monkeypatch, tmp_path: Path, status: int):
    monkeypatch.setattr(fetch_cmd.httpx, "stream", _Refusing(status))

    with pytest.raises(ExtractDownloadError):
        fetch_cmd._download_tile("https://example.invalid/t.tif", tmp_path / "t.tif")


def test_a_transport_error_is_a_download_failure(monkeypatch, tmp_path: Path):
    def timing_out(*args: object, **kwargs: object) -> None:
        raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(fetch_cmd.httpx, "stream", timing_out)

    with pytest.raises(ExtractDownloadError):
        fetch_cmd._download_tile("https://example.invalid/t.tif", tmp_path / "t.tif")


def test_no_partial_file_survives_a_refusal(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(fetch_cmd.httpx, "stream", _Refusing(HTTPStatus.NOT_FOUND))
    destination = tmp_path / "t.tif"

    with pytest.raises(DemTileUnpublishedError):
        fetch_cmd._download_tile("https://example.invalid/t.tif", destination)

    assert list(tmp_path.iterdir()) == []


def test_an_unpublished_square_is_written_down_for_later_builds(monkeypatch, tmp_path: Path):
    def unpublished(url: str, destination: Path) -> None:
        raise DemTileUnpublishedError(f"No tile published at {url}")

    monkeypatch.setattr(fetch_cmd, "_download_tile", unpublished)

    result = runner.invoke(
        fetch_cmd.app, ["dem", "--bbox", _ONE_TILE_BBOX, "--dest", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert load_unpublished(tmp_path) == frozenset({_ONE_TILE.name})


def test_a_failed_fetch_is_neither_recorded_nor_reported_as_success(monkeypatch, tmp_path: Path):
    def failing(url: str, destination: Path) -> None:
        raise ExtractDownloadError(f"Failed to download {url}: timed out")

    monkeypatch.setattr(fetch_cmd, "_download_tile", failing)

    result = runner.invoke(
        fetch_cmd.app, ["dem", "--bbox", _ONE_TILE_BBOX, "--dest", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert not registry_path(tmp_path).exists()
