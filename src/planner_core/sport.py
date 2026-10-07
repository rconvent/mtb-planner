"""What a sport contributes to the shared planner.

The engine — ingest, graph, elevation, search, export, API, interface — does
not know what it is planning for. Everything that does differ between a
mountain bike and a pair of trail shoes is gathered here, in one object each
sport package builds once:

* **Who may use a way, and in which direction.** `bicycle=no` closes a path to
  a rider and not to a runner; a one-way descent trail binds the bike only.
* **How hard a way is.** Both sports score technicality on the same 0–6
  ordinal so the cost model, the colour ramp and the profiles are shared, but
  each reads its own evidence onto it: `mtb:scale` for one, `sac_scale` for
  the other.
* **How fast the athlete covers ground.** A power balance for the bike, a
  metabolic cost of running for the runner. Both reduce to the same lookup
  table, so the search never asks which.
* **The words.** What the activity is called, what its difficulty grades look
  like, what to put in a GPX file name.

A process plans for one sport at a time. The entry point — `mtb` or `trail`,
or the app factory — activates it before anything else runs, and code that
needs it reads `active_sport()`. Engine functions that a test might want to
run under a different sport take it as an optional argument instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from planner_core.features.enums import SurfaceClass, TechConfidence
from planner_core.ingest.tags import WayTags

if TYPE_CHECKING:
    from planner_core.config.settings import Settings


class Access(IntEnum):
    """Whether, and how, a way may be used.

    The values are the ones the bike-only planner wrote to disk as `BikeAccess`,
    so artifacts built before the split still read correctly.
    """

    NO = 0
    #: Usable, but only at walking pace: a bike pushed up steps, a runner
    #: scrambling through terrain that needs hands.
    WALK = 1
    YES = 2


class Direction(IntEnum):
    """Permitted directions of travel along a way."""

    BOTH = 0
    FORWARD = 1
    BACKWARD = 2


@dataclass(frozen=True, slots=True)
class OnewayRule:
    """A resolved direction, and whether the evidence for it is trustworthy.

    A stray `oneway=yes` on a `highway=path` is a common mapping error, and
    enforcing it can silently make a whole area unloopable. Restrictions on
    paths are therefore only treated as hard when corroborated by a tag that
    implies deliberate directional design.
    """

    direction: Direction
    corroborated: bool


@dataclass(frozen=True, slots=True)
class Difficulty:
    """An edge's technicality, and how much is actually known about it."""

    tech: float
    tech_up: float
    confidence: TechConfidence
    flow: float


class Athlete(Protocol):
    """Whoever is covering the ground, as the time model needs them.

    `speed_table` is indexed `[grade band, surface class, technicality grade]`
    and returns metres per second. Evaluating it is the sport's whole speed
    model; everything downstream is a lookup.
    """

    @property
    def walk_speed_m_s(self) -> float:
        """Pace on ways marked `Access.WALK`."""
        ...

    def speed_table(self) -> npt.NDArray[np.float64]:
        """Speed for every grade band, surface and technicality."""
        ...


@dataclass(frozen=True)
class EffortSpec:
    """The one control that sets how hard the athlete goes.

    Watts for a rider, flat-ground pace for a runner. Requests carry it as a
    plain number called `effort`, and the sport says what that number means.
    """

    label: str
    unit: str
    min: float
    max: float
    step: float
    default: float


@dataclass(frozen=True)
class SportCopy:
    """The words a sport puts in front of people."""

    #: "Mountain bike route planner", as a page title or CLI help line.
    title: str
    #: One sentence for the page description and the API summary.
    summary: str
    #: The activity as a label: "Riding", "Running".
    activity: str
    #: The activity in a sentence: "riding", "running".
    activity_verb: str
    #: Prefix on a difficulty grade, so 2 reads "S2" or "T2".
    grade_prefix: str
    #: The scale the grades come from, named in the interface.
    scale_name: str
    #: Default track name in an exported GPX file.
    track_name: str


@dataclass(frozen=True)
class Sport:
    """Everything sport-specific, in one place."""

    #: Short machine name, recorded in every artifact this sport builds.
    id: str
    #: The command-line entry point, used in messages that suggest a command.
    cli: str
    #: Bumped whenever this sport's features or time model change in a way
    #: that alters routing results. Checked against every artifact at load.
    cost_model_version: int

    access: Callable[[WayTags], Access]
    direction: Callable[[WayTags], OnewayRule]
    difficulty: Callable[[WayTags, SurfaceClass], Difficulty]

    #: Route relation types read at ingest.
    route_types: frozenset[str]
    #: The subset that counts as a signed route for this sport. The rest are
    #: carried as connectors.
    signed_route_types: frozenset[str]

    #: The OSM key whose explicit value means "surveyed difficulty", and how to
    #: read it onto the shared scale. Used by the coverage report.
    surveyed_tag: str
    parse_surveyed: Callable[[str | None], float | None]
    #: The mode-specific access key (`bicycle`, `foot`), for the same report.
    access_tag: str
    #: Tags reported individually by `inspect`, because the difficulty model
    #: leans on them.
    headline_tags: tuple[str, ...]

    effort: EffortSpec
    athlete: Callable[[float | None], Athlete]

    profiles_path: Path
    default_profile: str
    #: Import string of this sport's ASGI app, for uvicorn — which needs one
    #: rather than an object when it reloads in a child process.
    asgi_app: str
    settings_cls: type[Settings]
    copy: SportCopy

    def is_relevant(self, tags: WayTags) -> bool:
        """Whether a way should enter this sport's routing graph at all."""
        return self.access(tags) is not Access.NO


_active: Sport | None = None


def activate(sport: Sport) -> Sport:
    """Make `sport` the one this process plans for.

    Clears the settings cache, because settings are read from a sport-specific
    environment prefix and a value cached under the previous sport would be
    answering for the wrong one.
    """
    global _active
    _active = sport

    from planner_core.config.settings import get_settings

    get_settings.cache_clear()
    return sport


def active_sport() -> Sport:
    """The sport this process plans for."""
    if _active is None:
        msg = (
            "No sport is active. Start the planner through `mtb` or `trail`, or call "
            "planner_core.sport.activate() first."
        )
        raise RuntimeError(msg)
    return _active
