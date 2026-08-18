"""Request and response models for planning.

Every range is enforced here so nothing downstream re-checks it. Coverage
checks are separate: they need the loaded artifact's bounding box, which the
schema layer has no business knowing about.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mtb_planner.api.schemas.common import LatLonModel
from mtb_planner.cost.profiles import FEATURE_NAMES, RideProfile
from mtb_planner.graph.store import GraphStore
from mtb_planner.ingest.tags import MAX_MTB_SCALE
from mtb_planner.routing.candidates import LoopMetrics
from mtb_planner.routing.result import Route
from mtb_planner.routing.segments import elevation_profile, route_segments

#: Sensible bounds on rider effort, in watts.
MIN_POWER_W = 80.0
MAX_POWER_W = 400.0

#: Coordinates are rounded on the way out. Beyond six decimals is under a
#: tenth of a metre — noise that only inflates the payload.
COORDINATE_DIGITS = 6


class SliderSpec(BaseModel):
    """One adjustable control, for a client to render without hard-coding it."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    min: float
    max: float
    step: float


class ProfileSummary(BaseModel):
    """A ride profile as offered to a client."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    tech_target: float
    tech_max: float
    detour: float

    @classmethod
    def of(cls, profile: RideProfile) -> ProfileSummary:
        """Summarise a loaded profile."""
        return cls(
            id=profile.id,
            name=profile.name,
            description=profile.description,
            tech_target=profile.tech_target,
            tech_max=profile.tech_max,
            detour=profile.detour,
        )


class ProfileList(BaseModel):
    """Everything a client needs to build the controls."""

    model_config = ConfigDict(frozen=True)

    profiles: tuple[ProfileSummary, ...]
    sliders: tuple[SliderSpec, ...]
    features: tuple[str, ...] = FEATURE_NAMES


class PlanRequest(BaseModel):
    """A point-to-point planning request."""

    model_config = ConfigDict(frozen=True)

    start: LatLonModel
    end: LatLonModel
    profile: str = Field(default="flow", min_length=1)
    tech_target: float | None = Field(default=None, ge=0.0, le=MAX_MTB_SCALE)
    detour: float | None = Field(default=None, ge=0.0, le=3.0)
    power_w: float | None = Field(default=None, ge=MIN_POWER_W, le=MAX_POWER_W)


#: Bounds on a loop's time budget, in minutes. Below the lower bound there is
#: not enough time to leave and return; above the upper one the search ball
#: covers a whole country.
MIN_BUDGET_MINUTES = 15
MAX_BUDGET_MINUTES = 600

#: How many loops a single request may ask for.
MAX_VARIANTS = 6


class LoopRequest(BaseModel):
    """A request for loops from a point within a time budget."""

    model_config = ConfigDict(frozen=True)

    start: LatLonModel
    budget_minutes: int = Field(ge=MIN_BUDGET_MINUTES, le=MAX_BUDGET_MINUTES)
    profile: str = Field(default="flow", min_length=1)
    variants: int = Field(default=3, ge=1, le=MAX_VARIANTS)
    tech_target: float | None = Field(default=None, ge=0.0, le=MAX_MTB_SCALE)
    detour: float | None = Field(default=None, ge=0.0, le=3.0)
    power_w: float | None = Field(default=None, ge=MIN_POWER_W, le=MAX_POWER_W)
    seed: int | None = Field(
        default=None,
        description=(
            "Accepted and echoed so a reported route can be quoted back exactly. "
            "Generation is deterministic — the same request returns the same loops "
            "whether a seed is given or not — so this currently changes nothing."
        ),
    )

    @property
    def budget_s(self) -> float:
        """The budget in seconds, which is what the generator works in."""
        return float(self.budget_minutes) * 60.0


class RouteStatsModel(BaseModel):
    """What the ride costs, measured rather than inferred."""

    model_config = ConfigDict(frozen=True)

    distance_m: float
    distance_km: float
    ascent_m: float
    descent_m: float
    moving_time_s: float
    moving_time_min: float
    average_speed_km_h: float
    dismount_m: float
    n_edges: int


class Geometry(BaseModel):
    """A GeoJSON LineString."""

    model_config = ConfigDict(frozen=True)

    type: str = "LineString"
    coordinates: list[tuple[float, float]]


class SnapModel(BaseModel):
    """Where a requested point attached to the network."""

    model_config = ConfigDict(frozen=True)

    point: LatLonModel
    distance_m: float


class SegmentModel(BaseModel):
    """One edge of a route, described by index into its geometry.

    Indices rather than coordinates: a route already ships its polyline once,
    and a second copy per segment would multiply the payload for nothing. A
    client slices `geometry.coordinates[first : last + 1]`.
    """

    model_config = ConfigDict(frozen=True)

    first: int
    last: int
    at_m: float
    tech: float
    surveyed: bool
    """Whether this segment's difficulty was mapped or inferred.

    Rendered as a distinct dash pattern. At a few percent survey coverage,
    drawing inferred difficulty identically to surveyed difficulty would assert
    something the data cannot support.
    """


class ElevationPointModel(BaseModel):
    """Height relative to the start, at a distance along the route.

    Relative, not altitude: the artifact stores per-edge ascent and descent but
    no absolute height, so there is no datum to anchor to. The shape of the
    climbing is what a profile chart draws, and this is consistent with the
    reported total ascent because it is summed from the same figures.
    """

    model_config = ConfigDict(frozen=True)

    at_m: float
    height_m: float


class CharacterModel(BaseModel):
    """What kind of riding a loop is, as fractions of its length.

    `unsurveyed_fraction` is a credibility field, not a nicety. At a few percent
    survey coverage, a difficulty figure presented without saying how much of it
    was inferred would be dishonest, so it travels with the difficulty it
    qualifies.
    """

    model_config = ConfigDict(frozen=True)

    technicality: float
    singletrack_fraction: float
    signed_fraction: float
    paved_fraction: float
    traffic: float
    unsurveyed_fraction: float


class RouteModel(BaseModel):
    """One planned route."""

    model_config = ConfigDict(frozen=True)

    route_id: str
    geometry: Geometry
    stats: RouteStatsModel
    segments: tuple[SegmentModel, ...] = ()
    elevation: tuple[ElevationPointModel, ...] = ()
    start_snap: SnapModel
    end_snap: SnapModel
    gpx_url: str
    warnings: tuple[str, ...] = ()

    # Present only for a loop. A point-to-point route is not ranked against
    # anything, so it has no character summary, no overlap and no budget verdict.
    character: CharacterModel | None = None
    overlap_ratio: float | None = None
    score: float | None = None
    within_budget: bool | None = None


class PlanResponse(BaseModel):
    """The result of a planning request."""

    model_config = ConfigDict(frozen=True)

    routes: tuple[RouteModel, ...]


class LoopModel(BaseModel):
    """One generated loop."""

    model_config = ConfigDict(frozen=True)

    route_id: str
    geometry: Geometry
    stats: RouteStatsModel
    character: CharacterModel
    segments: tuple[SegmentModel, ...] = ()
    elevation: tuple[ElevationPointModel, ...] = ()
    start_snap: SnapModel
    gpx_url: str
    overlap_ratio: float
    score: float
    within_budget: bool
    warnings: tuple[str, ...] = ()


class LoopResponse(BaseModel):
    """The result of a loop request."""

    model_config = ConfigDict(frozen=True)

    loops: tuple[LoopModel, ...]


def character_of(metrics: LoopMetrics) -> CharacterModel:
    """Convert a candidate's measurements for the wire."""
    return CharacterModel(
        technicality=round(metrics.tech, 2),
        singletrack_fraction=round(metrics.singletrack, 3),
        signed_fraction=round(metrics.signed, 3),
        paved_fraction=round(metrics.paved, 3),
        traffic=round(metrics.traffic, 3),
        unsurveyed_fraction=round(metrics.unknown, 3),
    )


def geometry_of(route: Route) -> Geometry:
    """Convert a route's polyline to GeoJSON, which is longitude first."""
    return Geometry(
        coordinates=[
            (round(float(lon), COORDINATE_DIGITS), round(float(lat), COORDINATE_DIGITS))
            for lat, lon in route.geometry
        ]
    )


def segments_of(store: GraphStore, route: Route) -> tuple[SegmentModel, ...]:
    """Convert a route's per-edge difficulty for the wire."""
    return tuple(
        SegmentModel(
            first=segment.first,
            last=segment.last,
            at_m=round(segment.at_m, 1),
            tech=round(segment.tech, 2),
            surveyed=segment.surveyed,
        )
        for segment in route_segments(store, route)
    )


def elevation_of(store: GraphStore, route: Route) -> tuple[ElevationPointModel, ...]:
    """Convert a route's relative height profile for the wire."""
    return tuple(
        ElevationPointModel(at_m=round(point.at_m, 1), height_m=round(point.height_m, 1))
        for point in elevation_profile(route_segments(store, route))
    )


def stats_of(route: Route) -> RouteStatsModel:
    """Convert a route's measurements for the wire."""
    stats = route.stats
    return RouteStatsModel(
        distance_m=round(stats.distance_m, 1),
        distance_km=round(stats.distance_km, 2),
        ascent_m=round(stats.ascent_m, 1),
        descent_m=round(stats.descent_m, 1),
        moving_time_s=round(stats.moving_time_s, 1),
        moving_time_min=round(stats.moving_time_min, 1),
        average_speed_km_h=round(stats.average_speed_km_h, 1),
        dismount_m=round(stats.dismount_m, 1),
        n_edges=stats.n_edges,
    )
