"""OSM tag capture and value parsing.

Only the ~30 keys the routing model uses are retained — the union of what
every sport reads, since a bike and a runner care about different ones; carrying whole tag
dictionaries through a multi-million-way build is the difference between a
graph that fits in memory and one that does not.

The parsers here are deliberately forgiving. Real OSM contains `mtb:scale=2+`,
`surface=gravel;dirt`, and `width=about 1m`. Every one of those must degrade to
a defined fallback rather than raise — a single bad tag should never abort a
build over an entire country.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

#: Tag keys carried through the build. Anything outside this set is dropped at
#: parse time and is invisible to everything downstream.
RETAINED_TAG_KEYS: frozenset[str] = frozenset(
    {
        "highway",
        "surface",
        "tracktype",
        "smoothness",
        "width",
        "mtb:scale",
        "mtb:scale:uphill",
        "mtb:scale:imba",
        "mtb:type",
        "sac_scale",
        "trail_visibility",
        "informal",
        "bicycle",
        "access",
        "foot",
        "motor_vehicle",
        "oneway",
        "oneway:bicycle",
        "oneway:mtb",
        "oneway:foot",
        "cycleway",
        "cycleway:left",
        "cycleway:right",
        "junction",
        "bridge",
        "tunnel",
        "layer",
        "incline",
        "service",
        "area",
        "route",
        "name",
        "ref",
        "step_count",
    }
)


@dataclass(frozen=True, slots=True)
class WayTags:
    """The retained tags of a single OSM way."""

    highway: str | None = None
    surface: str | None = None
    tracktype: str | None = None
    smoothness: str | None = None
    width: str | None = None
    mtb_scale: str | None = None
    mtb_scale_uphill: str | None = None
    mtb_scale_imba: str | None = None
    mtb_type: str | None = None
    sac_scale: str | None = None
    trail_visibility: str | None = None
    informal: str | None = None
    bicycle: str | None = None
    access: str | None = None
    foot: str | None = None
    motor_vehicle: str | None = None
    oneway: str | None = None
    oneway_bicycle: str | None = None
    oneway_mtb: str | None = None
    oneway_foot: str | None = None
    cycleway: str | None = None
    cycleway_left: str | None = None
    cycleway_right: str | None = None
    junction: str | None = None
    bridge: str | None = None
    tunnel: str | None = None
    layer: str | None = None
    incline: str | None = None
    service: str | None = None
    area: str | None = None
    route: str | None = None
    name: str | None = None
    ref: str | None = None
    step_count: str | None = None

    @classmethod
    def from_osm(cls, tags: Mapping[str, str]) -> WayTags:
        """Build from a raw OSM tag mapping, keeping only the retained keys."""
        return cls(
            highway=tags.get("highway"),
            surface=tags.get("surface"),
            tracktype=tags.get("tracktype"),
            smoothness=tags.get("smoothness"),
            width=tags.get("width"),
            mtb_scale=tags.get("mtb:scale"),
            mtb_scale_uphill=tags.get("mtb:scale:uphill"),
            mtb_scale_imba=tags.get("mtb:scale:imba"),
            mtb_type=tags.get("mtb:type"),
            sac_scale=tags.get("sac_scale"),
            trail_visibility=tags.get("trail_visibility"),
            informal=tags.get("informal"),
            bicycle=tags.get("bicycle"),
            access=tags.get("access"),
            foot=tags.get("foot"),
            motor_vehicle=tags.get("motor_vehicle"),
            oneway=tags.get("oneway"),
            oneway_bicycle=tags.get("oneway:bicycle"),
            oneway_mtb=tags.get("oneway:mtb"),
            oneway_foot=tags.get("oneway:foot"),
            cycleway=tags.get("cycleway"),
            cycleway_left=tags.get("cycleway:left"),
            cycleway_right=tags.get("cycleway:right"),
            junction=tags.get("junction"),
            bridge=tags.get("bridge"),
            tunnel=tags.get("tunnel"),
            layer=tags.get("layer"),
            incline=tags.get("incline"),
            service=tags.get("service"),
            area=tags.get("area"),
            route=tags.get("route"),
            name=tags.get("name"),
            ref=tags.get("ref"),
            step_count=tags.get("step_count"),
        )

    @property
    def is_bridge(self) -> bool:
        """Whether the way is carried on a bridge.

        Bridges must not be sampled against the DEM — a 30 m raster read
        across a viaduct reports the valley floor, not the deck.
        """
        return self.bridge not in (None, "no")

    @property
    def is_tunnel(self) -> bool:
        """Whether the way runs through a tunnel."""
        return self.tunnel not in (None, "no")


# --------------------------------------------------------------- parsing

#: `mtb:scale=2`, `S2`, `2+`, `1-`. The optional suffix is a third of a grade.
_SCALE_RE = re.compile(r"^s?([0-6])\s*([+-])?$")

#: `mtb:scale=1-2` — a range, not a modifier. Resolved to its midpoint.
_SCALE_RANGE_RE = re.compile(r"^s?([0-6])\s*-\s*s?([0-6])$")

#: `width=1.5`, `1,5 m`, `80 cm`. Anything with words in it is rejected.
_WIDTH_RE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(m|metre|metres|meter|meters|cm)?$")

#: A modifier shifts a grade by a third, matching how riders read the scale.
_MODIFIER_STEP = 1.0 / 3.0

#: Widths beyond this are typographic accidents, not trails.
_MAX_PLAUSIBLE_WIDTH_M = 20.0

MAX_MTB_SCALE = 6.0
MAX_IMBA_SCALE = 4.0


def parse_mtb_scale(value: str | None) -> float | None:
    """Parse an `mtb:scale` value to a float on the 0–6 Singletrail-Skala.

    Handles the plain grade, the `S` prefix, the `+`/`-` modifiers, and ranges
    such as `1-2`. A range is resolved to its midpoint; anything unrecognised
    returns `None` so the caller falls back to inference rather than guessing.

    Note the ambiguity in `1-`: a trailing dash is a modifier, while `1-2` is a
    range. They are distinguished by whether a digit follows the dash.
    """
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None

    if (match := _SCALE_RANGE_RE.match(text)) is not None:
        low, high = float(match.group(1)), float(match.group(2))
        return (low + high) / 2.0

    if (match := _SCALE_RE.match(text)) is not None:
        grade = float(match.group(1))
        modifier = match.group(2)
        if modifier == "+":
            grade += _MODIFIER_STEP
        elif modifier == "-":
            grade -= _MODIFIER_STEP
        return min(max(grade, 0.0), MAX_MTB_SCALE)

    return None


def parse_imba_scale(value: str | None) -> float | None:
    """Parse an `mtb:scale:imba` value to a float on the 0–4 IMBA scale."""
    if value is None:
        return None
    text = value.strip().lower().removeprefix("imba")
    if not text.isdigit():
        return None
    grade = float(text)
    return grade if grade <= MAX_IMBA_SCALE else None


def parse_width_m(value: str | None) -> float | None:
    """Parse a `width` value to metres.

    Accepts a bare number, a comma decimal separator, and an explicit `m` or
    `cm` unit. Rejects prose (`about 1m`), non-positive values, and anything
    implausibly wide.
    """
    if value is None:
        return None
    text = value.strip().lower().replace(" ", "")
    match = _WIDTH_RE.match(text)
    if match is None:
        return None

    width = float(match.group(1).replace(",", "."))
    if match.group(2) == "cm":
        width /= 100.0
    if width <= 0.0 or width > _MAX_PLAUSIBLE_WIDTH_M:
        return None
    return width


def parse_layer(value: str | None) -> int:
    """Parse a `layer` value, defaulting to ground level on anything odd."""
    if value is None:
        return 0
    try:
        return int(value.strip())
    except ValueError:
        return 0


def first_semicolon_value(value: str | None) -> str | None:
    """Return the first value of a semicolon-separated OSM list, lower-cased.

    `surface=gravel;dirt` is common and means "mostly the first one". Taking
    the head is closer to reality than discarding the tag entirely.
    """
    if value is None:
        return None
    head = value.split(";", 1)[0].strip().lower()
    return head or None
