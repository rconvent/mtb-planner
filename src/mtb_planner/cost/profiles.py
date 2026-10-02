"""Ride profiles: named presets over one weight vector.

There is deliberately no per-profile code path. A profile is a set of weights
over the same edge features, which means a slider overriding one weight
behaves identically whichever profile it started from, and adding a profile is
a configuration change rather than a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mtb_planner.core.errors import BuildError
from mtb_planner.ingest.tags import MAX_MTB_SCALE

DEFAULT_PROFILES_PATH = Path("config/profiles.yaml")

#: The greediest a search may be about trading time for better ground. It is
#: both the cap on a slider override and the ceiling the point-to-point slack
#: search bisects towards, so the two can never disagree about what "as far as
#: I am willing to go" means. Beyond this the route stops resembling a way of
#: getting anywhere.
MAX_DETOUR = 6.0

#: Every feature a profile may weight. A weight for anything else is a typo,
#: and silently ignoring it would leave the rider wondering why the slider did
#: nothing.
FEATURE_NAMES: tuple[str, ...] = (
    "tech",
    "singletrack",
    "signed_mtb",
    "flow",
    "unpaved",
    "fireroad",
    "paved",
    "traffic",
    "descent",
    "climb",
    "unknown",
)


class ProfileSpec(BaseModel):
    """One profile as written in the configuration file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    tech_target: float = Field(ge=0.0, le=MAX_MTB_SCALE)
    tech_sigma: float = Field(gt=0.0)
    tech_max: float = Field(ge=0.0, le=MAX_MTB_SCALE)
    detour: float = Field(ge=0.0, le=MAX_DETOUR)
    weights: dict[str, float]


@dataclass(frozen=True)
class RideProfile:
    """A resolved profile, ready to weight edges with."""

    id: str
    name: str
    description: str
    tech_target: float
    tech_sigma: float
    tech_max: float
    detour: float
    weights: tuple[float, ...]

    @classmethod
    def from_spec(cls, spec: ProfileSpec) -> RideProfile:
        """Resolve a configured profile, ordering weights by feature."""
        unknown = set(spec.weights) - set(FEATURE_NAMES)
        if unknown:
            msg = f"Profile {spec.id!r} weights unknown feature(s): {', '.join(sorted(unknown))}"
            raise BuildError(msg)
        return cls(
            id=spec.id,
            name=spec.name,
            description=spec.description.strip(),
            tech_target=spec.tech_target,
            tech_sigma=spec.tech_sigma,
            tech_max=spec.tech_max,
            detour=spec.detour,
            weights=tuple(spec.weights.get(name, 0.0) for name in FEATURE_NAMES),
        )

    def weight_of(self, feature: str) -> float:
        """The weight this profile puts on one named feature."""
        return self.weights[FEATURE_NAMES.index(feature)]

    def with_overrides(
        self,
        tech_target: float | None = None,
        detour: float | None = None,
        weights: dict[str, float] | None = None,
    ) -> RideProfile:
        """Apply slider overrides, leaving the original untouched.

        The sliders a rider sees are exactly these: what technicality they
        want, how far they will detour for it, and how much each feature
        matters.
        """
        adjusted = list(self.weights)
        for feature, value in (weights or {}).items():
            if feature not in FEATURE_NAMES:
                msg = f"Unknown feature {feature!r}; expected one of {', '.join(FEATURE_NAMES)}"
                raise BuildError(msg)
            adjusted[FEATURE_NAMES.index(feature)] = value

        return replace(
            self,
            tech_target=self.tech_target if tech_target is None else tech_target,
            detour=self.detour if detour is None else detour,
            weights=tuple(adjusted),
        )


def load_profiles(path: Path = DEFAULT_PROFILES_PATH) -> dict[str, RideProfile]:
    """Read every profile, keyed by id."""
    if not path.exists():
        msg = f"No ride profiles at {path}"
        raise BuildError(msg)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        msg = f"{path} is not valid YAML: {exc}"
        raise BuildError(msg) from exc

    entries = raw.get("profiles")
    if not entries:
        msg = f"{path} defines no profiles"
        raise BuildError(msg)

    try:
        specs = [ProfileSpec.model_validate(entry) for entry in entries]
    except ValidationError as exc:
        msg = f"{path} is not a usable profile set: {exc}"
        raise BuildError(msg) from exc

    profiles = {spec.id: RideProfile.from_spec(spec) for spec in specs}
    if len(profiles) != len(specs):
        msg = f"{path} defines duplicate profile ids"
        raise BuildError(msg)
    return profiles


def get_profile(profile_id: str, path: Path = DEFAULT_PROFILES_PATH) -> RideProfile:
    """Look up one profile by id, naming the alternatives if it is absent."""
    profiles = load_profiles(path)
    if profile_id not in profiles:
        available = ", ".join(sorted(profiles))
        msg = f"Unknown profile {profile_id!r}. Available: {available}"
        raise BuildError(msg)
    return profiles[profile_id]
