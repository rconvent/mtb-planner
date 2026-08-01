"""The application's error vocabulary.

Every failure the API can surface is one of these. `api/errors.py` maps each to
an HTTP status and a response envelope; nothing else in the codebase should
raise bare `Exception` or `ValueError` across a module boundary.
"""

from __future__ import annotations


class MtbPlannerError(Exception):
    """Base class for every error this application raises deliberately."""


# --------------------------------------------------------------- build time


class BuildError(MtbPlannerError):
    """Raised while fetching, parsing, or assembling the routing artifact."""


class ExtractDownloadError(BuildError):
    """A Geofabrik extract or DEM tile could not be fetched or verified."""


class PbfParseError(BuildError):
    """An `.osm.pbf` file could not be read, or produced no usable ways."""


class DemTileMissingError(BuildError):
    """A DEM tile covering part of the AOI is unavailable.

    Fail loudly: silently accepting a missing tile yields a flat graph, and
    flat terrain is indistinguishable from correct output until a rider
    notices the ascent figures are nonsense.
    """


# --------------------------------------------------------------- artifact


class ArtifactError(MtbPlannerError):
    """The on-disk routing artifact is missing, unreadable, or incompatible."""


class ArtifactNotFoundError(ArtifactError):
    """No artifact directory at the configured path."""


class IncompatibleArtifactError(ArtifactError):
    """Artifact schema or cost-model version does not match this build.

    Loading anyway would produce plausible-looking wrong routes with no error,
    which is worse than refusing to start.
    """


# --------------------------------------------------------------- request time


class PlanningError(MtbPlannerError):
    """Base class for failures while answering a routing request."""


class GraphNotLoadedError(PlanningError):
    """The routing graph is not resident yet, or failed to load."""


class PointOutsideCoverageError(PlanningError):
    """A requested point falls outside the artifact's bounding box."""

    def __init__(self, lat: float, lon: float) -> None:
        super().__init__(f"Point {lat:.5f},{lon:.5f} is outside the routable area")
        self.lat = lat
        self.lon = lon


class PointOffNetworkError(PlanningError):
    """A requested point is too far from any routable way.

    Carries the nearest routable point so the caller can offer to snap to it
    rather than just reporting failure.
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        nearest_lat: float | None = None,
        nearest_lon: float | None = None,
        distance_m: float | None = None,
    ) -> None:
        detail = f"No routable way near {lat:.5f},{lon:.5f}"
        if distance_m is not None:
            detail += f" (nearest is {distance_m:.0f} m away)"
        super().__init__(detail)
        self.lat = lat
        self.lon = lon
        self.nearest_lat = nearest_lat
        self.nearest_lon = nearest_lon
        self.distance_m = distance_m


class UnknownProfileError(PlanningError):
    """The requested ride profile does not exist.

    A separate type from `BuildError`, which the profile loader raises: a typo
    in a request is the client's mistake and belongs in the 4xx range, while a
    malformed `profiles.yaml` is a deployment fault and must not be disguised
    as one.
    """


class NoRouteFoundError(PlanningError):
    """No route satisfied the request.

    `reason` is machine-readable so the UI can suggest a specific fix rather
    than a generic failure message.
    """

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(detail or reason)
        self.reason = reason


class RouteExpiredError(PlanningError):
    """A previously planned route is no longer in the cache."""


class ServerBusyError(PlanningError):
    """Too many concurrent planning requests in flight."""
