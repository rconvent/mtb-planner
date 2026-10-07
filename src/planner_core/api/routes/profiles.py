"""The profiles on offer, and the controls that adjust them.

Answered from configuration alone, so it works before the graph has finished
loading — the controls should be visible while the artifact is still being
read, not an error.
"""

from __future__ import annotations

from fastapi import APIRouter

from planner_core.api.schemas.common import ApiResponse
from planner_core.api.schemas.planning import ProfileList, ProfileSummary, SliderSpec
from planner_core.cost.profiles import MAX_DETOUR, load_profiles
from planner_core.features.enums import TECH_SCALE_MAX
from planner_core.sport import active_sport

router = APIRouter(tags=["planning"])


def sliders() -> tuple[SliderSpec, ...]:
    """The controls a client should offer.

    Declared by the server so a front end does not hard-code them and drift
    from what the cost model actually reads. Effort is the sport's: watts for a
    rider, pace for a runner.
    """
    effort = active_sport().effort
    return (
        SliderSpec(id="tech_target", label="Technicality", min=0.0, max=TECH_SCALE_MAX, step=0.5),
        SliderSpec(id="detour", label="Willingness to detour", min=0.0, max=MAX_DETOUR, step=0.1),
        SliderSpec(
            id="effort",
            label=f"{effort.label} ({effort.unit})",
            min=effort.min,
            max=effort.max,
            step=effort.step,
        ),
    )


@router.get("/api/profiles", response_model=ApiResponse[ProfileList])
def list_profiles() -> ApiResponse[ProfileList]:
    """The profiles on offer, and the controls that adjust them."""
    profiles = load_profiles()
    return ApiResponse.ok(
        ProfileList(
            profiles=tuple(ProfileSummary.of(p) for p in profiles.values()),
            default_profile=active_sport().default_profile,
            sliders=sliders(),
        )
    )
