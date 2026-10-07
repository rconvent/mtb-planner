"""Reading `oneway`-style tag values.

Shared because every sport has a mode-specific key — `oneway:bicycle`,
`oneway:foot` — and all of them take the same values as `oneway` itself.
"""

from __future__ import annotations

from planner_core.ingest.tags import first_semicolon_value
from planner_core.sport import Direction

_ONEWAY_FORWARD: frozenset[str] = frozenset({"yes", "true", "1"})
_ONEWAY_BACKWARD: frozenset[str] = frozenset({"-1", "reverse"})
_ONEWAY_NONE: frozenset[str] = frozenset({"no", "false", "0"})


def direction_from_value(value: str | None) -> Direction | None:
    """Map a raw `oneway`-style value onto a direction, if it is meaningful."""
    normalised = first_semicolon_value(value)
    if normalised is None:
        return None
    if normalised in _ONEWAY_FORWARD:
        return Direction.FORWARD
    if normalised in _ONEWAY_BACKWARD:
        return Direction.BACKWARD
    if normalised in _ONEWAY_NONE:
        return Direction.BOTH
    return None
