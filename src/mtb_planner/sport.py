"""The mountain bike, as the shared planner sees it."""

from __future__ import annotations

from pathlib import Path

from mtb_planner import COST_MODEL_VERSION
from mtb_planner.access import resolve_access, resolve_direction
from mtb_planner.difficulty import estimate_difficulty
from mtb_planner.settings import MtbSettings
from mtb_planner.speed import Rider
from planner_core.ingest.tags import parse_mtb_scale
from planner_core.sport import Athlete, EffortSpec, Sport, SportCopy

#: Sensible bounds on rider effort, in watts.
MIN_POWER_W = 80.0
MAX_POWER_W = 400.0


def rider_for(power_w: float | None) -> Athlete:
    """A rider at this power, or the default rider."""
    return Rider() if power_w is None else Rider(power_w=power_w)


MTB = Sport(
    id="mtb",
    cli="mtb",
    cost_model_version=COST_MODEL_VERSION,
    access=resolve_access,
    direction=resolve_direction,
    difficulty=estimate_difficulty,
    # `mtb` is high-value signed riding; `bicycle` covers the knooppunten
    # networks, useful mainly as connectors.
    route_types=frozenset({"mtb", "bicycle"}),
    signed_route_types=frozenset({"mtb"}),
    surveyed_tag="mtb:scale",
    parse_surveyed=parse_mtb_scale,
    access_tag="bicycle",
    headline_tags=(
        "mtb:scale",
        "mtb:scale:uphill",
        "mtb:scale:imba",
        "mtb:type",
        "sac_scale",
        "tracktype",
        "surface",
        "smoothness",
        "width",
        "trail_visibility",
        "informal",
        "incline",
        "oneway",
        "oneway:bicycle",
    ),
    effort=EffortSpec(
        label="Effort",
        unit="watts",
        min=MIN_POWER_W,
        max=MAX_POWER_W,
        step=10.0,
        default=Rider().power_w,
    ),
    athlete=rider_for,
    profiles_path=Path("config/profiles/mtb.yaml"),
    default_profile="flow",
    asgi_app="mtb_planner.api:app",
    settings_cls=MtbSettings,
    copy=SportCopy(
        title="Mountain bike route planner",
        summary=(
            "Plan mountain bike loops and point-to-point routes against a time budget, "
            "scored on trail character."
        ),
        activity="Riding",
        activity_verb="riding",
        grade_prefix="S",
        scale_name="mtb:scale",
        track_name="MTB route",
    ),
)
