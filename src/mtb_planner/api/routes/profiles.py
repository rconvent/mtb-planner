"""The ride profiles on offer, and the controls that adjust them.

Answered from configuration alone, so it works before the graph has finished
loading — a rider should see the controls while the artifact is still being
read, not an error.
"""

from __future__ import annotations

from fastapi import APIRouter

from mtb_planner.api.schemas.common import ApiResponse
from mtb_planner.api.schemas.planning import (
    MAX_POWER_W,
    MIN_POWER_W,
    ProfileList,
    ProfileSummary,
    SliderSpec,
)
from mtb_planner.cost.profiles import load_profiles
from mtb_planner.ingest.tags import MAX_MTB_SCALE

router = APIRouter(tags=["planning"])

#: The controls a client should offer. Declared by the server so a front end
#: does not hard-code them and drift from what the cost model actually reads.
SLIDERS: tuple[SliderSpec, ...] = (
    SliderSpec(id="tech_target", label="Technicality", min=0.0, max=MAX_MTB_SCALE, step=0.5),
    SliderSpec(id="detour", label="Willingness to detour", min=0.0, max=3.0, step=0.1),
    SliderSpec(id="power_w", label="Effort (watts)", min=MIN_POWER_W, max=MAX_POWER_W, step=10.0),
)


@router.get("/api/profiles", response_model=ApiResponse[ProfileList])
def list_profiles() -> ApiResponse[ProfileList]:
    """The ride profiles on offer, and the controls that adjust them."""
    profiles = load_profiles()
    return ApiResponse.ok(
        ProfileList(
            profiles=tuple(ProfileSummary.of(p) for p in profiles.values()),
            sliders=SLIDERS,
        )
    )
