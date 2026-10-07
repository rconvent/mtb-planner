"""Geometry primitives.

Coordinates are WGS84 degrees at the boundary and scaled `int32` (1e-7 degrees)
inside the graph. Everything here is pure and vectorised where it matters —
the graph build samples tens of millions of points through these functions.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from planner_core.core.units import COORD_SCALE, EARTH_RADIUS_M


class LatLon(NamedTuple):
    """A WGS84 coordinate in degrees."""

    lat: float
    lon: float

    def to_scaled(self) -> tuple[int, int]:
        """Encode as the graph's `int32` 1e-7-degree pair."""
        return (round(self.lat * COORD_SCALE), round(self.lon * COORD_SCALE))

    @classmethod
    def from_scaled(cls, lat_e7: int, lon_e7: int) -> LatLon:
        """Decode from the graph's `int32` 1e-7-degree pair."""
        return cls(lat_e7 / COORD_SCALE, lon_e7 / COORD_SCALE)


class BBox(NamedTuple):
    """An axis-aligned bounding box in WGS84 degrees."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def contains(self, point: LatLon) -> bool:
        """Whether `point` falls inside the box, edges inclusive."""
        return (
            self.min_lat <= point.lat <= self.max_lat and self.min_lon <= point.lon <= self.max_lon
        )

    def expanded(self, degrees: float) -> BBox:
        """Return a copy grown by `degrees` on every side."""
        return BBox(
            self.min_lon - degrees,
            self.min_lat - degrees,
            self.max_lon + degrees,
            self.max_lat + degrees,
        )


def haversine_m(a: LatLon, b: LatLon) -> float:
    """Great-circle distance between two points, in metres."""
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def haversine_m_array(
    lat1: npt.NDArray[np.float64],
    lon1: npt.NDArray[np.float64],
    lat2: npt.NDArray[np.float64],
    lon2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Vectorised haversine over arrays of degrees, in metres.

    Used to measure polyline segment lengths during the graph build, where a
    per-point Python call would dominate build time.
    """
    rlat1, rlat2 = np.radians(lat1), np.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = np.radians(lon2 - lon1)
    h = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    distance: npt.NDArray[np.float64] = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(h))
    return distance


def polyline_length_m(points: npt.NDArray[np.float64]) -> float:
    """Total length of an (N, 2) array of `[lat, lon]` degrees, in metres."""
    if points.shape[0] < 2:
        return 0.0
    seg = haversine_m_array(points[:-1, 0], points[:-1, 1], points[1:, 0], points[1:, 1])
    return float(seg.sum())


def bearing_deg(a: LatLon, b: LatLon) -> float:
    """Initial bearing from `a` to `b`, in degrees clockwise from north.

    Used to spread loop pivots across heading sectors, which is the main
    reason generated loops look different from one another.
    """
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlon = math.radians(b.lon - a.lon)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def bearing_deg_array(
    lat0: float,
    lon0: float,
    lat: npt.NDArray[np.float64],
    lon: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Initial bearings from one point to many, in degrees clockwise from north.

    The loop generator buckets several thousand candidate turning points into
    heading sectors on every request, which is a poor use of a Python loop.
    """
    rlat0 = np.radians(lat0)
    rlat = np.radians(lat)
    dlon = np.radians(lon - lon0)
    y = np.sin(dlon) * np.cos(rlat)
    x = np.cos(rlat0) * np.sin(rlat) - np.sin(rlat0) * np.cos(rlat) * np.cos(dlon)
    bearings: npt.NDArray[np.float64] = np.degrees(np.arctan2(y, x)) % 360.0
    return bearings


def heading_sector(bearing: float, n_sectors: int) -> int:
    """Bucket a bearing into one of `n_sectors` equal sectors starting at north."""
    if n_sectors < 1:
        msg = "n_sectors must be at least 1"
        raise ValueError(msg)
    return int((bearing % 360.0) / (360.0 / n_sectors)) % n_sectors


def densify(points: npt.NDArray[np.float64], max_spacing_m: float) -> npt.NDArray[np.float64]:
    """Insert intermediate points so no segment exceeds `max_spacing_m`.

    Used before DEM sampling: Copernicus GLO-30 is ~30 m per pixel, so ~25 m
    spacing samples roughly every pixel. Finer spacing only adds noise.

    Interpolation is linear in degrees, which is accurate enough at these
    distances and far cheaper than a geodesic solution.
    """
    if points.shape[0] < 2 or max_spacing_m <= 0:
        return points

    seg_len = haversine_m_array(points[:-1, 0], points[:-1, 1], points[1:, 0], points[1:, 1])
    n_extra = np.maximum(1, np.ceil(seg_len / max_spacing_m).astype(np.int64))

    out: list[npt.NDArray[np.float64]] = []
    for i, steps in enumerate(n_extra):
        frac = np.linspace(0.0, 1.0, steps + 1)[:-1].reshape(-1, 1)
        out.append(points[i] + frac * (points[i + 1] - points[i]))
    out.append(points[-1].reshape(1, 2))
    return np.vstack(out)
