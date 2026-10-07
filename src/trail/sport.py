"""The trail runner, as the shared planner sees it."""

from __future__ import annotations

from pathlib import Path

from planner_core.sport import Athlete, EffortSpec, Sport, SportCopy
from trail import COST_MODEL_VERSION
from trail.access import resolve_access, resolve_direction
from trail.difficulty import estimate_difficulty, parse_sac_scale
from trail.settings import TrailSettings
from trail.speed import Runner

#: Bounds on flat-ground pace, in minutes per kilometre: from a strong club
#: runner to a run-walk.
FASTEST_PACE_MIN_KM = 3.5
SLOWEST_PACE_MIN_KM = 10.0


def runner_for(pace_min_km: float | None) -> Athlete:
    """A runner holding this flat pace, or the default runner."""
    return Runner() if pace_min_km is None else Runner(flat_pace_min_km=pace_min_km)


TRAIL = Sport(
    id="trail",
    cli="trail",
    cost_model_version=COST_MODEL_VERSION,
    access=resolve_access,
    direction=resolve_direction,
    difficulty=estimate_difficulty,
    # Waymarked walking routes are what a runner follows on the ground: GR and
    # regional paths are `hiking`, local promenades `foot`, the few signed
    # running circuits `running` or `fitness_trail`.
    route_types=frozenset({"hiking", "foot", "running", "fitness_trail"}),
    signed_route_types=frozenset({"hiking", "foot", "running", "fitness_trail"}),
    surveyed_tag="sac_scale",
    parse_surveyed=parse_sac_scale,
    access_tag="foot",
    headline_tags=(
        "sac_scale",
        "trail_visibility",
        "mtb:scale",
        "surface",
        "smoothness",
        "tracktype",
        "width",
        "informal",
        "incline",
        "foot",
        "oneway:foot",
    ),
    effort=EffortSpec(
        label="Flat pace",
        unit="min/km",
        min=FASTEST_PACE_MIN_KM,
        max=SLOWEST_PACE_MIN_KM,
        step=0.25,
        default=Runner().flat_pace_min_km,
    ),
    athlete=runner_for,
    profiles_path=Path("config/profiles/trail.yaml"),
    default_profile="trail",
    asgi_app="trail.api:app",
    settings_cls=TrailSettings,
    copy=SportCopy(
        title="Trail running route planner",
        summary=(
            "Plan trail running loops and point-to-point routes against a time budget, "
            "scored on trail character."
        ),
        activity="Running",
        activity_verb="running",
        grade_prefix="T",
        scale_name="hiking scale (sac_scale)",
        track_name="Trail run",
    ),
)
